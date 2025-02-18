import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_sparse import SparseTensor

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, DEFAULT_DIM, init_log, empty_cache

from GAT_DGL import GAT_DGL
from GAT_PyG import GAT_PyG
import dgl

from ctypes import cdll
cdll.LoadLibrary('/home/ljq/mine/graphiler/src/build/sputnik/libsputnik.so')
from GAT_MY import MyGAT, MyGAT_new, SputnikGAT


device = setup()

BREAK_FLAG = 2

USE_DGL_DATASET = True

if USE_DGL_DATASET:
    from graphiler.utils import homo_dataset
else:
    import sys
    sys.path.append("/home/ljq/mine/graphiler/examples")
    from AGNN import dataset
    from graphiler.utils import prepare_graph
    homo_dataset = {
                    'citeseer':3703, 'cora':1433, 'pubmed':500, 
                    'ppi':50, 'PROTEINS_full':29, 'OVCAR-8H':66,
                    'Yeast':74, 
                    'DD':89, 'YeastH':75, 'amazon0505':96,
                    'artist':100, 'com-amazon':96, 'soc-BlogCatalog':128,
                    'amazon0601':96}

# Currently Graphiler do not support full module compilation
# therefore, we pass extra parameters as a workaround for class member
# e.g., self.fc_weight, compare with GATLayer.message_func for the difference
def message_func(edges: EdgeBatchDummy, fc_weight, attn_weight):
    z_s = torch.mm(edges.src['h'], fc_weight)
    z_d = torch.mm(edges.dst['h'], fc_weight)
    z2 = torch.cat([z_s, z_d], dim=1)
    a = torch.mm(z2, attn_weight)
    return {'z': z_s, 'e': F.leaky_relu_(a)}


def reduce_func(nodes: NodeBatchDummy):
    alpha = torch.softmax(nodes.mailbox['e'], dim=1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h': h}


mpdfg = mpdfg_builder(message_func, reduce_func)
mpdfg_compile = mpdfg_builder(message_func, reduce_func, opt_level=0)
mpdfg_plus_reorder = mpdfg_builder(message_func, reduce_func, opt_level=1)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc_weight = torch.rand(in_dim, out_dim).to(device)
        self.attn_weight = torch.rand(2 * out_dim, 1).to(device)

    def message_func(self, edges):
        z_s = torch.mm(edges.src['h'], self.fc_weight)
        z_d = torch.mm(edges.dst['h'], self.fc_weight)
        z2 = torch.cat([z_s, z_d], dim=1)
        a = torch.mm(z2, self.attn_weight)
        return {'z': z_s, 'e': torch.relu(a)}

    def forward(self, g, feature, compile=False):
        g.ndata['h'] = feature
        if compile:
            if BREAK_FLAG == 0:
                update_all(g, mpdfg_compile, msg_params=(
                    self.fc_weight, self.attn_weight))
            elif BREAK_FLAG == 1:
                update_all(g, mpdfg_plus_reorder, msg_params=(
                    self.fc_weight, self.attn_weight))
            else:
                update_all(g, mpdfg, msg_params=(
                    self.fc_weight, self.attn_weight))
        else:
            g.update_all(self.message_func, reduce_func)
        return g.ndata.pop('h')


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, out_dim)

    def forward(self, g, features, compile=False):
        h = self.layer1(g, features, compile)
        h = F.elu(h)
        h = self.layer2(g, h, compile)
        return h

import matplotlib.pyplot as plt

def profile(dataset_name, feat_dim, repeat=1000):
    log = init_log(["0-DGL-UDF", "1-DGL-primitives", "2-PyG-primitives",
                    "3-Graphiler"], ["time", "mem"])
    print("benchmarking on: " + dataset_name)
    g, features = None, None
    if USE_DGL_DATASET:
        g, features = load_data(dataset_name, feat_dim, prepare=False)
    else:
        dir_path = os.path.dirname(__file__)
        g = dataset.TCGNN_dataset(os.path.join(dir_path, "../AGNN/tcgnn-ae-graphs/", dataset_name + ".npz"), feat_dim, load_from_txt=False)
        features = g.x
    features = features.to(device)

    @empty_cache
    def run_baseline_graphiler(g, features):
        if USE_DGL_DATASET:
            g, _ = load_data(dataset_name, feat_dim, prepare=True)
        else:
            g = dgl.graph(('csr', (g.row_pointers.to(torch.int64), 
                    g.column_index.to(torch.int64), [])), num_nodes=g.num_nodes)
            g = prepare_graph(g)
            g.ntype_data = {}
            g.etype_data = {}
        # print(g.ntypes, g.etypes)
        g = g.to(device)
        net = GAT(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                  out_dim=DEFAULT_DIM).to(device)
        net.eval()
        with torch.no_grad():
            compile_res = bench(net=net, net_params=(
                g, features, True), tag="3-Graphiler", nvprof=False, repeat=repeat, memory=True, log=log)
            # res = bench(net=net, net_params=(g, features, False),
            #             tag="0-DGL-UDF", nvprof=False, repeat=repeat, memory=True, log=log)
            # check_equal(compile_res, res)
        del g, net, compile_res

    @empty_cache
    def run_pyg(g, features):
        adj = None
        if USE_DGL_DATASET:
            u, v = g.edges()
            adj = SparseTensor(row=u, col=v, sparse_sizes=(
                g.num_src_nodes(), g.num_dst_nodes())).to(device)
        else:
            adj = SparseTensor(rowptr=g.row_pointers.to(torch.int64), col=g.column_index.to(torch.int64), 
                           sparse_sizes=(g.num_nodes, g.num_nodes)).to(device)
        
        net_pyg = GAT_PyG(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_pyg.eval()
        with torch.no_grad():
            bench(net=net_pyg, net_params=(features, adj),
                  tag="2-PyG-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
        del adj, net_pyg

    @empty_cache
    def run_dgl(g, features):
        if not USE_DGL_DATASET:
            g = dgl.graph(('csr', (g.row_pointers.to(torch.int64), 
                    g.column_index.to(torch.int64), 
                    [])), num_nodes=g.num_nodes) 
        g_ = g.to(device)
        net_dgl = GAT_DGL(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_dgl.eval()
        with torch.no_grad():
            bench(net=net_dgl, net_params=(g_, features),
                  tag="1-DGL-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
        del g_, net_dgl

    import mygraph
    @empty_cache
    def run_mygat(g, features):
        adj, node_num = None, None
        if USE_DGL_DATASET:
            u, v = g.edges()
            node_num = g.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
        else:
            adj = torch.IntTensor(g.edge_index).contiguous()
            node_num = g.num_nodes.item(0)
        adj_ = adj.clone()
        # counts = g.in_degrees(g.nodes(None)).tolist()
        # from minhash_order import reorder_weighted_minhashLSH
        # node_idx, dev_idx = reorder_weighted_minhashLSH(v, u, node_num)
        # print(node_idx, dev_idx, len(node_idx), len(dev_idx))
        node_idx = torch.arange(0, node_num).type(torch.IntTensor).to(device)
        dev_idx = torch.arange(0, node_num).type(torch.IntTensor).to(device)
        RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX = mygraph.process_DTC(adj.to(device), dev_idx, 16, 8, node_num, True)
        # RowWindowOffset_, RowWindowRowOffset_, TCOffset_, BlockMask_, SparseAToX_ = mygraph.process_DTC(adj.to(device), dev_idx, 16, 8, node_num, False)
        # assert(torch.all(RowWindowOffset[RowWindowRowOffset] == RowWindowOffset_))
        # print(RowWindowOffset[:10], RowWindowRowOffset[:10], TCOffset[:10])
        # print(RowWindowOffset, TCOffset, BlockMask, SparseAToX)
        # print(BlockMask[RowWindowOffset[161]*16:RowWindowOffset[162]*16])

        # group = 64
        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.scatter(u, v, s=0.1)
        # adj = torch.vstack([u, v]).type(torch.IntTensor)
        # new_adj, new_ord = adj, torch.arange(0, len(counts)).type(torch.IntTensor)
        # offset, out_edge_index = mygraph.preprocess_graph(adj.to(device), counts, group, features.size(0))
        # new_adj, new_ord = mygraph.reorder(adj)
        # density = new_adj.size()[1] / (new_ord.size()[0] * new_ord.size()[0])
        # print(new_ord.size()[0], density)
        # idx_set, idx_mask, idx_offset = mygraph.get_graph_set(new_adj.to(device), group, new_ord.size()[0])
        # density = idx_mask.sum().item() / (idx_mask.size()[0] * group)
        # print(density)
        # print(idx_set, idx_mask, idx_offset)
        # import os
        # os._exit(0)
        # print(new_ord, len(new_ord))
        # counts = counts[new_ord]
        # offset, out_edge_idx = mygraph.preprocess_graph(new_adj.to(device), counts.to(device), 1, features.size(0))
        # plt.subplot(222)
        # plt.scatter(new_adj, new_adj, s=0.1)
        # plt.show()

        # plt.scatter(u, v[:500])
        # plt.show()
        # map = torch.arange(0, len(counts)).type(torch.IntTensor).to(device)
        # counts = torch.cumsum(torch.cat([torch.tensor([0]), torch.asarray(g.in_degrees(g.nodes(None).tolist()))]), dim=0).type(torch.IntTensor).to(device)
        # out = g.in_edges(g.nodes(None), form='eid').type(torch.IntTensor).to(device)
        net_gat = MyGAT(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                        out_dim=DEFAULT_DIM).to(device)
        net_gat.eval()
        with torch.no_grad():
            bench(net=net_gat, net_params=(features, adj_.to(device), RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX),
                  tag='4-MyGat-primitives', nvprof=False, repeat=repeat, memory=True, log=log)
        del net_gat, adj, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX, adj_, node_idx, dev_idx

    @empty_cache
    def run_sputnik_gat(g, features):
        adj, node_num, counts = None, None, None
        if USE_DGL_DATASET:
            u, v = g.edges()
            node_num = g.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor).to(device)
            counts = g.in_degrees(g.nodes(None)).tolist()
        else:
            adj = torch.IntTensor(g.edge_index).contiguous().to(device)
            node_num = g.num_nodes.item(0)
            counts = g.degrees
        counts = torch.tensor(counts, dtype=torch.int32, device=device)
        row_offset, edge_idx = mygraph.preprocess_CSR(adj, counts, 1, node_num)
        adj_ = adj[:, edge_idx].contiguous()
        net_gat = SputnikGAT(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                        out_dim=DEFAULT_DIM).to(device)
        net_gat.eval()
        with torch.no_grad():
            bench(net=net_gat, net_params=(features, adj_.to(device), row_offset),
                  tag='5-SputnikGat-primitives', nvprof=False, repeat=repeat, memory=True, log=log)
        del net_gat, adj, row_offset, edge_idx, adj_, counts

    @empty_cache
    def run_mygat_new(g, features):
        adj, node_num = None, None
        if USE_DGL_DATASET:
            u, v = g.edges()
            node_num = g.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
        else: 
            adj = torch.IntTensor(g.edge_index).contiguous()
            node_num = g.num_nodes.item(0)
        RowWindowOffset, BitMaskRowOffset, BitColMask, BitRowMask, SparseAToX = mygraph.process_DTC_short_mask(adj.to(device), 16, 16, node_num, False)
        net_gat = MyGAT_new(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                        out_dim=DEFAULT_DIM, block_size=(16, 16)).to(device)
        net_gat.eval()
        with torch.no_grad():
            bench(net=net_gat, net_params=(features, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask),
                  tag='6-MyGat-new', nvprof=False, repeat=repeat, memory=True, log=log)
        del net_gat, adj, RowWindowOffset, BitMaskRowOffset, BitColMask, BitRowMask, SparseAToX


    # run_baseline_graphiler(g, features)
    # run_pyg(g, features)
    # run_mygat(g, features)
    # run_mygat_new(g, features)
    # run_dgl(g, features)
    run_sputnik_gat(g, features)
    

    return log

def breakdown(dataset, feat_dim, repeat=1000):
    log = init_log(['0-DGL-UDF', '1+compile', '2+reorder',
                   '3+fusion'], ['time', 'mem'])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = GAT(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
              out_dim=DEFAULT_DIM).to(device)
    net.eval()
    with torch.no_grad():
        bench(net=net, net_params=(g, features, False),
              tag="0-DGL-UDF", nvprof=False, repeat=repeat, memory=True, log=log)
        global BREAK_FLAG
        BREAK_FLAG = 0
        bench(net=net, net_params=(
            g, features, True), tag="1+compile", nvprof=False, repeat=repeat, memory=True, log=log)
        BREAK_FLAG = 1
        bench(net=net, net_params=(
            g, features, True), tag="2+reorder", nvprof=False, repeat=repeat, memory=True, log=log)
        BREAK_FLAG = 2
        bench(net=net, net_params=(
            g, features, True), tag="3+fusion", nvprof=False, repeat=repeat, memory=True, log=log)

    return log


if __name__ == '__main__':
    repeat = int(os.environ.get('REPEAT', 1000))
    if len(sys.argv) != 3:
        print("usage: python GAT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in homo_dataset:
            log[d] = profile(d, homo_dataset[d], repeat)
        pd.DataFrame(log).to_pickle("examples/GAT/output/GAT.csv")
    elif sys.argv[1] == "breakdown":
        log = {}
        for d in homo_dataset:
            log[d] = breakdown(d, homo_dataset[d], repeat)
        pd.DataFrame(log).to_pickle("output/GAT_breakdown.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat)
