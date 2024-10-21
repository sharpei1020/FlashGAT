import numpy as np
from datasketch import MinHash, MinHashLSH, WeightedMinHashGenerator, WeightedMinHash
from scipy.sparse import coo_matrix
import torch
import time
import random
import os
import os.path as osp
import argparse
import cugraph
import cudf
import libMHCUDA
import queue as Q

lsh_thres = 0.2
cluster_thres = 0.2
c_thres = 128
per = 128
thres = 16

def root(i, cluster_id):
    while i != cluster_id[i]:
        cluster_id[i] = cluster_id[cluster_id[i]]
        i = cluster_id[i]
    return i

def makenum(a, b, num_row):
    if a > b:
        tmp = a
        a = b
        b = tmp
    return a * num_row + b

def jd(l1,l2):
    if len(l1) == 0 or len(l2) == 0:
        return 0
    s1 = set(l1)
    s2 = set(l2)
    return (float)(len(s1.intersection(s2))) / len(s1.union(s2))

class Pair(object):
    def __init__(self,p1,p2,similarity):
        self.p1 = p1
        self.p2 = p2
        self.simi = similarity
    def __lt__(self,other): # operator < 
        return self.simi > other.simi
    def __str__(self):
        return str(self.p1) + ' ' + str(self.p2) + ' ' + str(self.simi)

def reorder_weighted_minhashLSH(edge_index_0, edge_index_1, node_num):
    num_nnz = len(edge_index_0)
    val = np.array([1] * num_nnz, dtype=np.float32)
    edge_index = np.stack((edge_index_0, edge_index_1))
    scipy_coo = coo_matrix((val, edge_index), shape=(node_num, node_num))
    scipy_csr = scipy_coo.tocsr()
    scipy_csr.data = scipy_csr.data.tolist()
    ptr = scipy_csr.indptr
    idx = scipy_csr.indices
    row_ind = np.array(edge_index_0)
    col_ind = np.array(edge_index_1)

    print("=== Init lsh ===")
    t0 = time.time()
    lsh = MinHashLSH(threshold=lsh_thres, num_perm=per)
    allver = []
    lists = [[] for i in range(node_num)]
    if node_num < 1600000:  # out of memory
        bgen = WeightedMinHashGenerator(node_num)
        gen = libMHCUDA.minhash_cuda_init(node_num, 128, seed=1, verbosity=1)
        libMHCUDA.minhash_cuda_assign_vars(gen, bgen.rs, bgen.ln_cs, bgen.betas)
        hashes = libMHCUDA.minhash_cuda_calc(gen, scipy_csr)
        libMHCUDA.minhash_cuda_fini(gen)
        for i in range(node_num):
            len_i = ptr[i+1] - ptr[i]
            m = WeightedMinHash(2022, hashes[i])
            lsh.insert(str(i), m)
            allver.append(m)
            for iter in range(ptr[i], ptr[i+1]):
                lists[i].append(idx[iter])
    else:
        for i in range(node_num):
            m = MinHash(num_perm=per)
            for iter in range((int)(ptr[i]), (int)(ptr[i+1])):
                m.update(str(idx[iter]).encode('utf-8'))
                lists[i].append(idx[iter])
            lsh.insert(str(i), m)
            allver.append(m)

    t1 = time.time()
    print("init LSH time (s)", t1 - t0)

    que = Q.PriorityQueue()
    sset = set()
    G = cugraph.Graph()
    df = cudf.DataFrame()
    df["src"] = cudf.Series(row_ind)
    df["dst"] = cudf.Series(col_ind)
    df["val"] = cudf.Series(val)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="val")

    print("=== TCU-Aware level clustering===")
    t2 = time.time()
    first = np.empty((0,), dtype=np.int64)
    second = np.empty((0,))
    for i in range(node_num):
        if ptr[i] == ptr[i + 1]:
            continue
        res = lsh.query(allver[i])
        first = np.concatenate((first, np.full(len(res), i, dtype=np.int64)))
        second = np.concatenate((second, np.array(res)))
        if i % 1000 == 0 or i == node_num - 1:
            print("reach row: ", i)
            df_pairs = cudf.DataFrame()
            df_pairs["first"] = first
            df_pairs["second"] = second
            s = time.time()
            df1 = cugraph.jaccard(G, df_pairs).to_pandas().values
            for j in range(df1.shape[0]):
                source =  (int)(df1[j][1])
                item = (int)(df1[j][2])
                if item == source or makenum(source, item, node_num) in sset:
                    continue
                que.put(Pair(source, item, df1[j][0]))
                sset.add(makenum(source, item, node_num))
            e = time.time()
            first = np.empty((0,))
            second = np.empty((0,))

    print("queue size: ", que.qsize())
    t3 = time.time()
    print("query LSH time (s): ", t3 - t2)

    cluster_id = [i for i in range(node_num)]
    cluster_sz = [1 for i in range(node_num)]
    deleted = [0 for i in range(node_num)]
    num_cluster = node_num

    t4 = time.time()
    while (not que.empty()) and num_cluster > 0:
        item = que.get()
        p1 = item.p1
        p2 = item.p2
        sset.remove(makenum(p1, p2, node_num))
        if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
            if deleted[p1] or deleted[p2]:
                continue
            if cluster_sz[p1] < cluster_sz[p2]:
                cluster_id[p1] = p2
                num_cluster = num_cluster - 1
                cluster_sz[p2] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p2] >= thres:
                    deleted[p2] = 1
                    num_cluster = num_cluster - 1
            else:
                cluster_id[p2] = p1
                num_cluster = num_cluster - 1
                cluster_sz[p1] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p1] >= thres:
                    deleted[p1] = 1
                    num_cluster = num_cluster - 1
        else:
            p1 = root(p1, cluster_id)
            p2 = root(p2, cluster_id)
            if deleted[p1] or deleted[p2]:
                continue
            if p1 != p2 and not makenum(p1, p2, node_num) in sset:
                que.put(Pair(p1, p2, jd(lists[p1], lists[p2])))
                sset.add(makenum(p1, p2, node_num))
    t5 = time.time()
    print("clustering time (s): ", t5 - t4)

    clusters = {}
    t6 = time.time()
    for i in range(node_num):
        ro = root(i, cluster_id)
        if ro in clusters:
            clusters[ro].append(i)
        else:
            clusters[ro] = [i]

    t7 = time.time()
    print("put into clusters time (s): ", t7 - t6)
    cluster_num = len(clusters)
    print("cluster_num:", cluster_num)

    print("=== Cache-Aware level clustering ===")
    key = list(clusters.keys())
    per_c = 128   # for 4090
    lsh_c = MinHashLSH(threshold=cluster_thres, num_perm=per_c)
    allver_c = []
    lists_c = [[] for i in range(cluster_num)]   # unique column indices for each cluster lists_c[i]: indices for cluster i
    cnt = 0
    for i in clusters:
        m = MinHash(num_perm=per_c)
        list_cluster_i = [] 
        for node in clusters[i]:
            list_cluster_i = list_cluster_i + lists[node]
        list_cluster_i = list(set(list_cluster_i))
        lists_c[cnt] = list_cluster_i
        for idx in list_cluster_i:
            m.update(str(idx).encode('utf-8'))
        lsh_c.insert(str(cnt), m)
        allver_c.append(m)
        cnt = cnt + 1
    que_c = Q.PriorityQueue()
    sset_c = set()
    t2 = time.time()
    for i in range(cluster_num):
        if i % 1000 == 0:
            print("reach cluster: ", i)
        if(len(lists_c[i])==0):
            continue
        res = lsh_c.query(allver_c[i])
        for item in res:
            if (int)(item) == i or makenum(i, (int)(item), cluster_num) in sset_c:
                continue
            if len(lists_c[(int)(item)]) == 0:
                continue
            que_c.put(Pair(i, (int)(item), jd(lists_c[i], lists_c[(int)(item)])))
            sset_c.add(makenum(i, (int)(item), cluster_num))
    print("cluster queue size:", que_c.qsize())
    t3 = time.time()
    print("query cluster LSH time (s): ", t3 - t2)
    cluster_id_c = [i for i in range(cluster_num)]
    cluster_sz_c = [1 for i in range(cluster_num)]
    deleted_c = [0 for i in range(cluster_num)]
    num_cluster_c = cluster_num
    t4 = time.time()
    while (not que_c.empty()) and num_cluster_c > 0:
        item = que_c.get()
        p1 = item.p1
        p2 = item.p2
        sset_c.remove(makenum(p1, p2, cluster_num))
        if p1 == cluster_id_c[p1] and p2 == cluster_id_c[p2]:
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if cluster_sz_c[p1] < cluster_sz_c[p2]:
                cluster_id_c[p1] = p2
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p2] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p2] >= c_thres:
                    deleted_c[p2] = 1
                    num_cluster_c = num_cluster_c - 1
            else:
                cluster_id_c[p2] = p1
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p1] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p1] >= c_thres:
                    deleted_c[p1] = 1
                    num_cluster_c = num_cluster_c - 1
        else:
            p1 = root(p1, cluster_id_c)
            p2 = root(p2, cluster_id_c)
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if p1 != p2 and not makenum(p1, p2, cluster_num) in sset_c:
                que_c.put(Pair(p1, p2, jd(lists_c[p1], lists_c[p2])))
                sset_c.add(makenum(p1, p2, cluster_num))
    t5 = time.time()
    print("cluster clustering time (s): ", t5 - t4)
    clusters_c = {}
    t6 = time.time()
    for i in range(cluster_num):
        ro = root(i, cluster_id_c)
        if ro in clusters_c:
            clusters_c[ro].append(i)
        else:
            clusters_c[ro] = [i]
    cluster_cluster_num = len(clusters_c)
    print("cluster_of_cluster_num: ", cluster_cluster_num)
    t7 = time.time()
    print("put clusters into clusters time (s): ", t7 - t6)

    print("=== Save results ===")
    node_idx = []
    for j in clusters_c:
        for k in clusters_c[j]:
            clustersk = clusters[key[k]]
            for item in clustersk:
                node_idx.append(item)
    dev_idx = sorted(range(node_num), key=lambda k: node_idx[k])
    # print("node_idx device:", node_idx.device, "dev_idx device:", dev_idx.device)
    return torch.tensor(node_idx, dtype=torch.int32, device=torch.device('cuda')), torch.tensor(dev_idx, dtype=torch.int32, device=torch.device('cuda'))




    
