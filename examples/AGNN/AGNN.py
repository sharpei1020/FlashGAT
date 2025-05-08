#!/usr/bin/env python3
import dgl.graph_index
from scipy.sparse import coo_matrix
import numpy as np
import torch
import torch.nn as nn
import sys
import math
import time 
import os

from tqdm.std import tqdm
import TCGNN
from AGNN_PyG import AGNN_PyG
from AGNN_DGL import AGNN_DGL
from ctypes import cdll
cdll.LoadLibrary('/home/ljq/mine/graphiler/src/build/sputnik/libsputnik.so')
from AGNN_MY import MyAGNN, SputnikAGNN, MyAGNN_new, MyAGNN, MyAGNNlayer_adaptive, MyAGNNlayer_csr
from AGNN_UDF import UDFAGNN
import pandas as pd
import mygraph

import dgl
from torch_sparse import SparseTensor
from graphiler.utils import init_log, load_data, setup, empty_cache, bench
import torch.nn.functional as F
from torch_geometric.utils import softmax

n_heads = 1
device = setup()

USE_DGL_DATASET = True

# BREAK_FLAG = 2

# def message_func(edges:)

if USE_DGL_DATASET:
    from graphiler.utils import homo_dataset
else:
    from dataset import *
    homo_dataset = {
                    'citeseer':3703, 'cora':1433, 'pubmed':500, 
                    'ppi':50, 'PROTEINS_full':29, 'OVCAR-8H':66,
                    'Yeast':74, 'DD':89, 'YeastH':75, 'amazon0505':96,
                    'artist':100, 'com-amazon':96, 'soc-BlogCatalog':128,
                    'amazon0601':96, 'web-BerkStan':100, 'web-Google':100,
                    'web-NotreDame':100, 'web-Stanford':100}

class TCGNNFunction_AGNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, orign):

        # ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        # X_prime = torch.mm(X, weights)
        x_prime = None
        if orign:
            X_prime = torch.mm(X, weights)
        else:
            X_prime = F.normalize(X, p=2, dim=-1)
        
        # SDDMM: edge feature computation. 
        # import time
        # torch.cuda.synchronize()
        # t = time.time()
        edge_feature = TCGNN.forward_ef(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # torch.cuda.synchronize()
        # print("sddmm time:{:.2f} ms".format((time.time() - t) * 1000))

        # Edge Attention Generation: [n_e, n_head]       
        edge_attentions = None
        if orign:
            edge_attentions = torch.mm(edge_feature.unsqueeze(-1), attention_w).transpose(0,1).contiguous() 
        else: 
            edge_attentions = torch.mm(edge_feature.unsqueeze(-1), attention_w).transpose(0,1).contiguous()
            edge_attentions = softmax(edge_attentions.squeeze(), edgeToRow.to(torch.int64), 
                                  row_pointers.to(torch.int64), num_nodes=len(row_pointers)-1).reshape(1, -1)
        # print(edge_attentions.size())
        # SpMM_AGNN: Neighbor AggreAGNNion.
        # torch.cuda.synchronize()
        # t = time.time()
        X_prime = TCGNN.forward_AGNN(X_prime, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)[0]
        # torch.cuda.synchronize()
        # print("spmm time:{:.2f} ms".format((time.time() - t) * 1000))
        # print("finish forward")
        ctx.save_for_backward(X, weights, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)
        # print("==========After Aggreation=========")
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward_AGNN(d_output, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)

        # attention weight back propaAGNNion.
        d_attention = TCGNN.forward_ef(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # print(d_attention.size())
        d_attention_exp = d_attention[None, :].expand(n_heads, -1)
        # print(d_attention_exp.size())

        d_attention_w = torch.mm(d_attention_exp, column_index[:, None].float()).transpose(0,1)
        # print(d_attention_w.size())

        return d_input, d_weights, d_attention_w, None, None, None, None, None
    

class AGNNConvLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConvLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_w = torch.nn.Parameter(torch.randn(1, n_heads))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, orign):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": X, "row_pointers": row_pointers, "column_index": column_index, 
        #           "blockPartition": blockPartition, "edgeToColumn": edgeToColumn, "edgeToRow": edgeToRow,
        #           "attention_w": self.attention_w}
        # # container = torch.jit.script(Container(mydata))
        # # torch.jit.save(container, f"TC_data.pt")
        # print(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, self.attention_w)
        # import os
        # os._exit(0)

        return TCGNNFunction_AGNN.apply(X, self.weights, self.attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, orign)


class TC_AGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, orign):
        super(TC_AGNN, self).__init__()
        self.lin1 = AGNNConvLayer(input_dim, hidden_dim) if orign else nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        layer_num = 2 if orign else 4
        for _ in range(layer_num):
            self.hidden_layers.append(AGNNConvLayer(hidden_dim, hidden_dim))
        self.lin2 = AGNNConvLayer(hidden_dim, output_dim) if orign else nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(True)
        self.orign = orign

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        X = self.relu(self.lin1(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, self.orign) if self.orign else self.lin1(X))
        for Gconv in self.hidden_layers:
            X = self.relu(Gconv(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, self.orign))
        X = self.lin2(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, self.orign) if self.orign else self.lin2(X)
        return X

DIM = 32

from enum import Enum
class udf_type(Enum):
    Base = 0
    SDDMM_TCGNN = 1
    SDDMM_MY = 2
    Softmax = 3
    SpMM = 4
    ALL = 5
    ALL_MINE = 6

def profile(dataset_name, feat_dim, repeat=1000):
    log = init_log(["1-PyG-primitives", "2-TCGNN-primitives", "3-mygraph", "4-DGL-primitives", "5-Sputnik-primitives"],
                   ["time", "mem"])
    print("benchmarking on: " + dataset_name)
    features, dataset = None, None
    if USE_DGL_DATASET:
        dataset, features = load_data(dataset_name, feat_dim, prepare=False)
        features = features.to(device)
    else:
        dataset = TCGNN_dataset(dataset_name, feat_dim, load_from_txt=False)    
        features = dataset.x.to(device)

    @empty_cache
    def run_sputnik(dataset, features):
        adj, node_num, u, v = None, None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            node_num = dataset.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
        else:
            adj = torch.IntTensor(dataset.edge_index).contiguous()
            node_num = dataset.num_nodes.item(0)
        self = torch.vstack([torch.arange(0, node_num).type(torch.IntTensor)] * 2).to(device)
        adj_ = torch.unique(torch.torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        row_idx, edge_idx = mygraph.process_CSR(adj_, 1, node_num)
        unique_row_idx, counts = torch.unique_consecutive(row_idx, return_counts=True)
        counts_ = torch.zeros(node_num, device=device, dtype=torch.int32)
        counts_[unique_row_idx] = counts.to(torch.int32)
        row_offset = torch.cumsum(counts_, 0)
        row_offset = torch.cat([torch.tensor([0]).to(device), row_offset]).to(torch.int32)
        adj_ = adj_[:, edge_idx].contiguous()
        net = SputnikAGNN(in_dim=feat_dim, hidden_dim=DIM,
                        out_dim=DIM).to(device)
        net.eval()
        with torch.no_grad():
            bench(net=net, net_params=(features, adj_, row_offset),
                  tag='5-Sputnik-primitives', nvprof=False, repeat=repeat, memory=True, log=log)
        del dataset, net, u, v, adj, row_offset, edge_idx, adj_, counts, counts_, unique_row_idx

    @empty_cache
    def run_dgl(dataset, features):
        g = None
        if USE_DGL_DATASET:
            g = dataset.to(device)
        else:
            g = dgl.graph(('csr', (dataset.row_pointers.to(torch.int64), 
                    dataset.column_index.to(torch.int64), 
                    [])), num_nodes=dataset.num_nodes).to(device)
        # print("get graph")
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5).to(device)
        g.ndata['norm_scheme'] = norm.unsqueeze(1)
        # print(g)
        net = AGNN_DGL(g, feat_dim, DIM, 4).to(device)
        net.eval()
        with torch.no_grad():
            bench(net=net, net_params=(features,), tag="4-DGL-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
        del g, degs, norm, net

    @empty_cache
    def run_pyg(dataset, features):
        adj = None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            adj = SparseTensor(row=u, col=v, sparse_sizes=(
                dataset.num_nodes(), dataset.num_nodes())).to(device)
        else:
            adj = SparseTensor(rowptr=dataset.row_pointers.to(torch.int64), col=dataset.column_index.to(torch.int64), 
                           sparse_sizes=(dataset.num_nodes, dataset.num_nodes)).to(device)
        net_pyg = AGNN_PyG(feat_dim, DIM).to(device)
        net_pyg.eval()
        with torch.no_grad():
            bench(net=net_pyg, net_params=(features, adj), 
                  tag="1-PyG-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
        del adj, net_pyg

    @empty_cache
    def run_tcgnn(dataset, features):
        num_nodes, num_edges, col_idx, row_ptr = None, None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            num_nodes = dataset.num_nodes()
            num_edges = len(u)
            val = np.array([1] * num_edges, dtype=np.float32)
            edge_index = np.stack((u, v))
            scipy_csr = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes)).tocsr()
            row_ptr = torch.tensor(scipy_csr.indptr)
            col_idx = torch.tensor(scipy_csr.indices)
        else:
            num_nodes = dataset.num_nodes
            num_edges = dataset.num_edges
            col_idx =  dataset.column_index 
            row_ptr = dataset.row_pointers
        
        num_row_windows = (num_nodes + 16 - 1) // 16
        edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
        edgeToRow = torch.zeros(num_edges, dtype=torch.int)
        blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
        # preprocessing for generating meta-information
        start = time.perf_counter()
        TCGNN.preprocess(col_idx, row_ptr, num_nodes,  \
                        16,	8, blockPartition, edgeToColumn, edgeToRow)
        
        row_ptr = row_ptr.to(device)
        col_idx = col_idx.to(device)
        edgeToColumn = edgeToColumn.to(device)
        edgeToRow = edgeToRow.to(device)
        blockPartition = blockPartition.to(device)

        build_neighbor_parts = time.perf_counter() - start
        print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))

        net = TC_AGNN(feat_dim, DIM, DIM, True).to(device)
        net_ = TC_AGNN(feat_dim, DIM, DIM, False).to(device)
        net.eval()
        net_.eval()
        with torch.no_grad():
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow),
                  tag="2-TCGNN-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
            bench(net=net_, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow),
                  tag="2-TCGNN-w", nvprof=False, repeat=repeat, memory=True, log=log)
        del num_nodes, num_edges, row_ptr, col_idx, num_row_windows, edgeToColumn, edgeToRow, blockPartition, net, net_
        
    @empty_cache
    def run_mygraph(dataset, features):
    # def run_mygraph(g, features):
        adj, node_num = None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            node_num = dataset.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
        else:
            adj = torch.IntTensor(dataset.edge_index).contiguous()
            node_num = dataset.num_nodes.item(0)
        # adj_ = torch.LongTensor(dataset.edge_index).contiguous()
        self = torch.vstack([torch.arange(0, node_num).type(torch.IntTensor)] * 2).to(device)
        adj_ = torch.unique(torch.torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        
        dev_idx = torch.arange(0, node_num).type(torch.IntTensor).to(device)
        time_start = time.perf_counter()

        # RowWindowOffsets, TCOffsets, BlockMasks, SparseAToXs = [], [], [], []
        RowWindowOffset, _, TCOffset, BlockMask, SparseAToX = mygraph.process_DTC(adj_.to(device), dev_idx, 16, 8, node_num, False)
            # RowWindowOffsets.append(RowWindowOffset)
            # TCOffsets.append(TCOffset)
            # BlockMasks.append(BlockMask)
            # SparseAToXs.append(SparseAToX)
        print("Prep. (ms):\t{:.3f}".format((time.perf_counter() - time_start)*1e3))
        # print(RowWindowOffset[:2], BlockMask[:RowWindowOffset[1]*16], TCOffset[:2])
        net = MyAGNN(feat_dim, DIM, DIM).to(device)
        net.eval()
        # with torch.no_grad():
        #     for i in range(50):
        #         net(adj, features, RowWindowOffsets[i%3], TCOffsets[i%3], BlockMasks[i%3], SparseAToXs[i%3])
        with torch.no_grad():
            bench(net=net, net_params=(adj.type(torch.LongTensor), features, RowWindowOffset, TCOffset, BlockMask, SparseAToX),
                            tag="3-mygraph", nvprof=False, repeat=repeat, memory=True, log=log)
        del adj, node_num, adj_, dev_idx, RowWindowOffset, TCOffset, BlockMask, SparseAToX, net

    @empty_cache
    def run_udf(dataset, features):
        num_nodes, num_edges, col_idx, row_ptr = None, None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            num_nodes = dataset.num_nodes()
            num_edges = len(u)
            val = np.array([1] * num_edges, dtype=np.float32)
            edge_index = np.stack((u, v))
            scipy_csr = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes)).tocsr()
            row_ptr = torch.tensor(scipy_csr.indptr)
            col_idx = torch.tensor(scipy_csr.indices)
        else:
            num_nodes = dataset.num_nodes
            num_edges = dataset.num_edges
            col_idx =  dataset.column_index 
            row_ptr = dataset.row_pointers
        
        num_row_windows = (num_nodes + 16 - 1) // 16
        edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
        edgeToRow = torch.zeros(num_edges, dtype=torch.int)
        blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
        TCGNN.preprocess(col_idx, row_ptr, num_nodes,  \
                         16,	8, blockPartition, edgeToColumn, edgeToRow) 
        row_ptr = row_ptr.to(device)
        col_idx = col_idx.to(device)
        edgeToColumn = edgeToColumn.to(device)
        edgeToRow = edgeToRow.to(device)
        blockPartition = blockPartition.to(device)

        RowWindow_offset, TCblock_offset, TCblocktile_id, sparseAToidx = mygraph.DTC_compression(
            row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow)
        RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, sparseAToX = mygraph.SGT_short_Mask(
            row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow, 16, 8)
        
        assert(RowWindow_offset.equal(RowWindowOffset))
        assert(sparseAToidx.equal(sparseAToX))
        
        net = UDFAGNN(feat_dim, DIM, DIM).to(device)
        net.eval()
############################################
        mid_edge_feature, mid_softmax = None, None
        out_feat = net.get_submodule("lin1")(features)
        x_prime = F.normalize(out_feat, p=2, dim=1)
        edge_feature = TCGNN.forward_ef(x_prime, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow)[0]
        mid_edge_feature = torch.mm(edge_feature.unsqueeze(-1), torch.ones((1, 1), device=device)).transpose(0,1).contiguous()
        mid_softmax = softmax(mid_edge_feature.squeeze(), edgeToRow.to(torch.int64), row_ptr.to(torch.int64), len(row_ptr)-1).reshape(1, -1)
############################################
        with torch.no_grad():
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
                RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
                BitMask_row, (udf_type.Base, None)), tag="UDF-base", nvprof=False, 
                repeat=repeat, memory=True, log=log)
            # bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
            #     RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
            #     BitMask_row, (udf_type.SDDMM_TCGNN, None)), tag="UDF-SDDMM_TCGNN", nvprof=False, 
            #     repeat=repeat, memory=True, log=log)
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
                RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
                BitMask_row, (udf_type.SDDMM_MY, None)), tag="UDF-SDDMM_MY", nvprof=False, 
                repeat=repeat, memory=True, log=log)
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
                RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
                BitMask_row, (udf_type.Softmax, mid_edge_feature)), tag="UDF-softmax", nvprof=False, 
                repeat=repeat, memory=True, log=log)
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
                RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
                BitMask_row, (udf_type.SpMM, mid_softmax)), tag="UDF-DTC_SpMM", nvprof=False, 
                repeat=repeat, memory=True, log=log)
            bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
                RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
                BitMask_row, (udf_type.ALL, None)), tag="UDF-ALL", nvprof=False, 
                repeat=repeat, memory=True, log=log)
            # bench(net=net, net_params=(features, row_ptr, col_idx, blockPartition, edgeToColumn, edgeToRow,
            #     RowWindow_offset, TCblocktile_id, TCblock_offset, sparseAToidx, BitMask_RowOffset, BitMask_col, 
            #     BitMask_row, (udf_type.ALL_MINE, None)), tag="UDF-ALL_MINE", nvprof=False, 
            #     repeat=repeat, memory=True, log=log)
        del num_nodes, num_edges, row_ptr, col_idx, num_row_windows, edgeToColumn, edgeToRow, blockPartition, \
        net, mid_edge_feature, mid_softmax, RowWindowOffset, BitMask_RowOffset, BitMask_col, BitMask_row, sparseAToX

    @empty_cache
    def run_myagnn_new(dataset, features):
        adj, node_num, adj_ = None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            node_num = dataset.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
            adj_ = torch.hstack([u, v]).type(torch.LongTensor).reshape(2, -1)
        else:
            adj = torch.IntTensor(dataset.edge_index).contiguous()
            node_num = dataset.num_nodes.item(0)
        self = torch.vstack([torch.arange(0, node_num).type(torch.IntTensor)] * 2).to(device)
        # adj_0 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        # adj_1 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        # adj_2 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        # adj_3 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        adj_4 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        adj_5 = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        # RowWindowOffset_, BitMaskRowOffset_, BitColMask_, BitRowMask_, SparseAToX_ = mygraph.process_DTC_short_mask(adj_0, 16, 8, node_num, False)
        # net_ = MyAGNN_new(feat_dim, DIM, DIM, (16, 8)).to(device)
        # RowWindowOffset, BitMaskRowOffset, BitColMask, BitRowMask, SparseAToX = mygraph.process_DTC_short_mask(adj_1, 16, 16, node_num, False)
        # net = MyAGNN_new(feat_dim, DIM, DIM, (16, 16)).to(device)
        # RowWindowOffset__, BitMaskRowOffset__, BitColMask__, BitRowMask__, SparseAToX__ = mygraph.process_DTC_short_mask(adj_2, 8, 16, node_num, False)
        # net__ = MyAGNN_new(feat_dim, DIM, DIM, (8, 16)).to(device)
        # RowWindowOffset_1, BitMaskRowOffset_1, BitColMask_1, BitRowMask_1, SparseAToX_1 = mygraph.process_DTC_short_mask(adj_3, 8, 8, node_num, False)
        # net_1 = MyAGNN_new(feat_dim, DIM, DIM, (8, 8)).to(device)
        RowWindowOffset_2, BitMaskRowOffset_2, BitColMask_2, BitRowMask_2, SparseAToX_2 = mygraph.process_DTC_short_mask(adj_4, 4, 8, node_num, False)
        net_2 = MyAGNN_new(feat_dim, DIM, DIM, (4, 8)).to(device)
        RowWindowOffset_3, BitMaskRowOffset_3, BitColMask_3, BitRowMask_3, SparseAToX_3 = mygraph.process_DTC_short_mask(adj_5, 2, 16, node_num, False)
        net_3 = MyAGNN_new(feat_dim, DIM, DIM, (2, 16)).to(device)
        # net_.eval()
        # net.eval()
        # net__.eval()
        # net_1.eval()
        net_2.eval()
        net_3.eval()
        with torch.no_grad():
            # bench(net=net_, net_params=(features, adj_, RowWindowOffset_, SparseAToX_, BitMaskRowOffset_, BitColMask_, BitRowMask_), 
            #         tag="3-myagnn_new-1608", nvprof=False, repeat=repeat, memory=True, log=log)
            # bench(net=net, net_params=(features, adj_, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask), 
            #         tag="3-myagnn_new-1616", nvprof=False, repeat=repeat, memory=True, log=log)
            # bench(net=net__, net_params=(features, adj_, RowWindowOffset__, SparseAToX__, BitMaskRowOffset__, BitColMask__, BitRowMask__), 
            #         tag="3-myagnn_new-816", nvprof=False, repeat=repeat, memory=True, log=log)
            # bench(net=net_1, net_params=(features, adj_, RowWindowOffset_1, SparseAToX_1, BitMaskRowOffset_1, BitColMask_1, BitRowMask_1), 
            #         tag="3-myagnn_new-808", nvprof=False, repeat=repeat, memory=True, log=log)
            bench(net=net_2, net_params=(features, adj_, RowWindowOffset_2, SparseAToX_2, BitMaskRowOffset_2, BitColMask_2, BitRowMask_2), 
                    tag="3-myagnn_new-408", nvprof=False, repeat=repeat, memory=True, log=log)
            bench(net=net_3, net_params=(features, adj_, RowWindowOffset_3, SparseAToX_3, BitMaskRowOffset_3, BitColMask_3, BitRowMask_3), 
                    tag="3-myagnn_new-216", nvprof=False, repeat=repeat, memory=True, log=log)
        del adj, node_num, \
        # adj_0, RowWindowOffset_, BitMaskRowOffset_, BitColMask_, BitRowMask_, SparseAToX_, net_,\
        # adj_1, RowWindowOffset, BitMaskRowOffset, BitColMask, BitRowMask, SparseAToX, net, \
        # adj_2, RowWindowOffset__, BitMaskRowOffset__, BitColMask__, BitRowMask__, SparseAToX__, net__,\
        # adj_3, RowWindowOffset_1, BitMaskRowOffset_1, BitColMask_1, BitRowMask_1, SparseAToX_1, net_1,\
        adj_4, RowWindowOffset_2, BitMaskRowOffset_2, BitColMask_2, BitRowMask_2, SparseAToX_2, net_2, \
        adj_5, RowWindowOffset_3, BitMaskRowOffset_3, BitColMask_3, BitRowMask_3, SparseAToX_3, net_3
    
    @empty_cache
    def run_myagnn_adaptive(dataset, features):
        adj, node_num, adj_ = None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            node_num = dataset.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
            adj_ = torch.hstack([u, v]).type(torch.LongTensor).reshape(2, -1)
        else:
            adj = torch.IntTensor(dataset.edge_index).contiguous()
            node_num = dataset.num_nodes.item(0)
        self = torch.vstack([torch.arange(0, node_num).type(torch.IntTensor)] * 2).to(device)
        adj = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        params = mygraph.adaptive_ASC(adj, "agnn", node_num, node_num)
        net = MyAGNN(feat_dim, DIM, DIM, MyAGNNlayer_adaptive).to(device)
        net.eval()
        with torch.no_grad():
            bench(net=net, net_params=(features, adj_, *params), tag="3-myagnn_adaptive", nvprof=False, repeat=repeat, memory=True, log=log)
        del params, adj, net

    @empty_cache
    def run_myagnn_csr(dataset, features):
        adj, node_num, adj_ = None, None, None
        if USE_DGL_DATASET:
            u, v = dataset.edges()
            node_num = dataset.num_nodes()
            adj = torch.vstack([u, v]).type(torch.IntTensor)
            adj_ = torch.hstack([u, v]).type(torch.LongTensor).reshape(2, -1)
        else:
            adj = torch.IntTensor(dataset.edge_index).contiguous()
            node_num = dataset.num_nodes.item(0)
        self = torch.vstack([torch.arange(0, node_num).type(torch.IntTensor)] * 2).to(device)
        adj = torch.unique(torch.hstack([self, adj.to(device)]).transpose(0,1), dim=0).transpose(0,1).contiguous()
        params = mygraph.process_CSR(adj.to(device), node_num)
        net = MyAGNN(feat_dim, DIM, DIM, MyAGNNlayer_csr).to(device)
        net.eval()
        with torch.no_grad():
            bench(net=net, net_params=(features, adj_, *params), tag="3-myagnn_csr", nvprof=False, repeat=repeat, memory=True, log=log)
        del params, adj, net

    # run_pyg(dataset, features)
    # run_tcgnn(dataset, features)
    # run_mygraph(dataset, features)
    # run_dgl(dataset, features)
    # run_sputnik(dataset, features)
    # # run_udf(dataset, features)
    # run_myagnn_new(dataset, features)
    run_myagnn_csr(dataset, features)
    # run_myagnn_adaptive(dataset, features)

    return log

if __name__ == '__main__':
    repeat = int(os.environ.get('REPEAT', 50))
    if len(sys.argv) != 3:
        print("usage: python GAT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        # homo_dataset = {'citeseer':3703, 'cora':1433, 'pubmed':500, 
        #                 'ppi':50, 'PROTEINS_full':29, 'OVCAR-8H':66,
        #                 'Yeast':74, 'DD':89, 'YeastH':75, 'amazon0505':96,
        #                 'artist':100, 'com-amazon':96, 'soc-BlogCatalog':128,
        #                 'amazon0601':96}
        for d in homo_dataset:
            log[d] = profile(d, homo_dataset[d], repeat)
        pd.DataFrame(log).to_csv("examples/AGNN/output/AGNN.csv")
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat)

