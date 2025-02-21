#!/usr/bin/env python3
import torch
import numpy as np
import time
import os

from scipy.sparse import *

def func(x):
    if x > 0:
        return x
    else:
        return 1
    
TCGNN_Data = {'citeseer':'citeseer.npz', 'cora':'cora.npz', 'pubmed':'pubmed.npz', 
    'ppi':'ppi.npz', 'PROTEINS_full':'PROTEINS_full.npz', 'OVCAR-8H':'OVCAR-8H.npz',
    'Yeast':'Yeast.npz', 'DD':'DD.npz', 'YeastH':'YeastH.npz', 'amazon0505':'amazon0505.npz',
    'artist':'artist.npz', 'com-amazon':'com-amazon.npz', 'soc-BlogCatalog':'soc-BlogCatalog.npz',
    'amazon0601':'amazon0601.npz', 'web-Stanford':'web-Stanford.txt', 'web-BerkStan':'web-BerkStan.txt',
    'web-Google':'web-Google.txt', 'web-NotreDame':'web-NotreDame.txt'}

class TCGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, dataset_name, dim, load_from_txt=True, verbose=False):
        super(TCGNN_dataset, self).__init__()

        self.nodes = set()

        self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        # self.num_classes = num_class
        self.edge_index = None
        
        self.reorder_flag = False
        self.verbose_flag = verbose

        self.avg_degree = -1
        self.avg_edgeSpan = -1

        dir_path = os.path.dirname(__file__)
        self.path = os.path.join(dir_path, "tcgnn-ae-graphs/", TCGNN_Data[dataset_name])
        self.init_edges()
        self.init_embedding(dim)
        

    def init_edges(self):

        # loading from a txt graph file
        if self.path.endswith('.txt'):
            # fp = open(self.path, "r")
            # src_li = []
            # dst_li = []
            # start = time.perf_counter()
            # for line in fp:
            #     print(line)
            #     src, dst = line.strip('\n').split()
            #     src, dst = int(src), int(dst)
            #     src_li.append(src)
            #     dst_li.append(dst)
            #     self.nodes.add(src)
            #     self.nodes.add(dst)
            
            self.edge_index = np.array(np.loadtxt(self.path), dtype=np.int32).reshape(2, -1)
            self.edge_index = self.edge_index if np.min(self.edge_index) == 0 else self.edge_index - 1
            # print(self.edge_index)
            self.num_edges = len(self.edge_index[0])
            self.num_nodes = np.max(self.edge_index) + 1
            # dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (txt) {:.3f}s ".format(dur))

        # loading from a .npz graph file
        elif self.path.endswith('.npz'): 
            start = time.perf_counter()
            graph_obj = np.load(self.path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']

            self.num_nodes = graph_obj['num_nodes']
            self.num_edges = len(src_li)
            self.edge_index = np.stack([src_li, dst_li])
            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (npz)(s): {:.3f}".format(dur))

        else:
            print("Error: Unknown file format.")
        
        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(self.edge_index[0], self.edge_index[1])))

        if self.verbose_flag:
            print('# nodes: {}'.format(self.num_nodes))
            print("# avg_degree: {:.2f}".format(self.avg_degree))
            print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

        # Build graph CSR.
        val = [1] * self.num_edges
        start = time.perf_counter()
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start

        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = degrees
    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    # def init_labels(self, num_class):
    #     '''
    #     Generate the node label.
    #     Called from __init__.
    #     '''
    #     self.y = torch.ones(self.num_nodes).long().cuda()
