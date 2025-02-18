import torch
from torch import Tensor
import torch.nn as nn
from typing import Any, Optional, List
import math
import copy
from torch_geometric.utils import scatter, softmax
import torch.nn.functional as F
import mygraph



def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

class SumAggregation(nn.Module):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, dim_size, dim, reduce='sum')
     
    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               dim_size: Optional[int] = None, dim: int = -2, reduce: str = 'sum') -> Tensor:

        assert index is not None
        return scatter(x, index.type(torch.int64), dim, dim_size, reduce)

class MyGATlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 concat=True,
                 dropout=0.,
                 activation=None):
        super(MyGATlayer, self).__init__()

        self.node_dim = -2
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.lin = nn.Linear(in_feats, num_heads * out_feats, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.att_j = nn.Parameter(torch.Tensor(1, num_heads, out_feats))

        self.reset_parameters()
        self.activation = activation
        self.aggr_module = SumAggregation()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX, layer_i):
############################################################
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": x, "RowWindowOffset": RowWindowOffset, "TCOffset": TCOffset, 
        #           "BlockMask": BlockMask, "SparseAToX": SparseAToX}
        
        # container = torch.jit.script(Container(mydata))
        # torch.jit.save(container, f"{layer_i}_data.pt")
        # import os
        # os._exit(0)

        # torch.jit.save(torch.jit.script(self.lin), f"{layer_i}_lin.pt")
        # torch.jit.save(torch.jit.script(Container({"att_i": self.att_i.data})), f"{layer_i}_att_i.pt")
        # torch.jit.save(torch.jit.script(Container({"att_j": self.att_j.data})), f"{layer_i}_att_j.pt")
        # x_test = torch.zeros((x.size(0), self._out_feats * self._num_heads)).to("cuda")
##################################################################
        # torch.backends.cudnn.allow_tf32 = False
        # if torch.is_tensor(x):
        #     # x_ = self.dropout(x)
        #     x_ = self.lin(x)
        #     x_ = (x_, x_)
        # else:
        #     x_ = (self.dropout(x[0]), self.dropout(x[1]))
        #     x_ = (self.lin(x_[0]), self.lin(x_[1]))

        # out_ = self.propagate(edge_index, x=x_)


        # x_test = torch.empty((x.size(0), self._out_feats), dtype=x.dtype, device="cuda")
        out = mygraph.gat_balance(x, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX, self.lin.weight,\
                          self.att_i.data, self.att_j.data, self._num_heads, self._out_feats)
############################################################        
        # condition_out = torch.all(torch.abs(out_ - out) < 0.4, dim=-1).cpu()
        # idxs = torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape)))
        # for i in range(16):
        #     print(out_[idxs[i]], out[idxs[i]])
        # debug_counts = torch.tensor(counts)[idxs].squeeze()[:16]
        # print(debug_counts)
        # for i in range(16):
        #     row_idx = idxs[i] // 16
        #     row_start = RowWindowRowOffset[row_idx]
        #     row_end = RowWindowRowOffset[row_idx+1]
        #     row_blocknum_set = []
        #     for j in range(row_start, row_end):
        #         row_blocknum_set.append((RowWindowOffset[j+1] - RowWindowOffset[j]).item())
        #     print(f"row {row_idx}: {row_blocknum_set}")

        # print(idxs[:16], len(idxs))
        # BlockMask.cpu()[RowWindowOffset[161]*16:RowWindowOffset[162]*16].reshape(-1, 16)
        # assert(torch.all(condition_out).item())
#############################################################
        # condition_lin = torch.all(torch.abs(lin_x - x_test) < 1e-5, dim=-1).cpu()
        # print(torch.nonzero(torch.where(condition_lin, torch.zeros(condition_lin.shape), torch.ones(condition_lin.shape))))
        # assert(torch.all(condition_lin).item())
###############################################################

        # out = self.propagate(edge_index, x=x_)

        # if self.activation is not None:
        #     out = self.activation(out)

        # if not self.concat:
        #     out = out.view(-1, self._num_heads, self._out_feats).mean(dim=1)
        
###############################################################
        # condition_out = torch.all(torch.abs(out - out_test) < 1e-5, dim=-1).cpu()
        # print(torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape))))
        # assert(torch.all(condition_out).item())
###############################################################

        return out

    def message(self, x_i, x_j, edge_index_i, size_i):
        # Compute attention coefficients.
        # torch.backends.cudnn.allow_tf32 = False

        x_i = x_i.view(-1, self._num_heads, self._out_feats)
        x_j = x_j.view(-1, self._num_heads, self._out_feats)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha)
######################################################################
        # condition_alpha = torch.all(torch.abs(alpha[:50] - alpha_test) < 1e-5, dim=-1).cpu()
        # print(torch.nonzero(torch.where(condition_alpha, torch.zeros(condition_alpha.shape), torch.ones(condition_alpha.shape))))
        # assert(torch.all(condition_alpha).item())
####################################################################
        alpha = softmax(alpha, edge_index_i.type(torch.int64), num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = self.dropout(alpha)

        rst = x_j * alpha.view(-1, self._num_heads, 1)
        return rst.view(-1, self._num_heads * self._out_feats)

    def propagate(self, edge_index: Tensor, x: Tensor):
        size = [None, None]
        data = self._collect(['x_i', 'x_j'], edge_index, size, x)
        out = self.message(data['x_i'], data['x_j'], 
                           data['edge_index_i'], data['size_i'])
        out = self.aggr_module(out, data['edge_index_i'], data['ptr'],
                              data['size_i'], self.node_dim)
        return self.update(out)
        
    def update(self, input: Tensor) -> Tensor:
        return input

    def _collect(self, args, edge_index, size, data_):
        i,j = (1, 0) 
        out = {}
        for arg in args:
            data = copy.deepcopy(data_)
            dim = j if arg[-2:] == '_j' else i
            if isinstance(data, (tuple, list)):
                assert len(data) == 2
                if isinstance(data[1 - dim], Tensor):
                    self._set_size(size, 1 - dim, data[1 - dim])
                data = data[dim]
            if isinstance(data, Tensor):
                self._set_size(size, dim, data)
                data = self._lift(data, edge_index, dim)

            out[arg] = data
        
        out['adj_t'] = None
        out['edge_index_i'] = edge_index[i]
        out['ptr'] = None
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']
        return out
    
    def _set_size(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))
        
    def _lift(self, src, edge_index, dim):
        index = edge_index[dim]
        return src.index_select(self.node_dim, index)

class MyGAT(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(MyGAT, self).__init__()

        self.conv1 = MyGATlayer(in_dim, hidden_dim, 1)
        self.conv2 = MyGATlayer(hidden_dim, out_dim, 1)

    def forward(self, x, adj, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX):
        h = self.conv1(x, adj, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX, 1)
        h = F.elu(h)
        h = self.conv2(h, adj, RowWindowOffset, RowWindowRowOffset, TCOffset, BlockMask, SparseAToX, 2)
        return h

class MyGATlayer_new(MyGATlayer):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 concat=True,
                 dropout=0.,
                 activation=None):
        super(MyGATlayer_new, self).__init__(in_feats,
                 out_feats,
                 num_heads)

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, RowWindowOffsets, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask, block_size):
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": x, "RowWindowOffset": RowWindowOffsets, "BitMaskRowOffset": BitMaskRowOffset, 
        #           "BitColMask": BitColMask, "BitRowMask": BitRowMask,"SparseAToX": SparseAToX}
        
        # container = torch.jit.script(Container(mydata))
        # torch.jit.save(container, f"data_16x16.pt")
        # import os
        # os._exit(0)
        return mygraph.gat_short(x, RowWindowOffsets, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask, self.lin.weight,\
                                  self.att_i.data, self.att_j.data, self._num_heads, self._out_feats, block_size[0], block_size[1])

class MyGAT_new(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim, block_size):
        super(MyGAT_new, self).__init__()
        self.block_size = block_size

        self.conv1 = MyGATlayer_new(in_dim, hidden_dim, 1)
        self.conv2 = MyGATlayer_new(hidden_dim, out_dim, 1)

    def forward(self, x, RowWindowOffsets, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask):
        h = self.conv1(x, RowWindowOffsets, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask, self.block_size)
        h = F.elu(h)
        h = self.conv2(h, RowWindowOffsets, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask, self.block_size)
        return h

class SputnikGATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 concat=True,
                 dropout=0.,
                 activation=None):
        super(SputnikGATLayer, self).__init__()

        self.node_dim = -2
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.lin = nn.Linear(in_feats, num_heads * out_feats, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.att_j = nn.Parameter(torch.Tensor(1, num_heads, out_feats))

        self.reset_parameters()
        self.activation = activation
        self.aggr_module = SumAggregation()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, row_id, row_ptr, col_id):
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": x, "row_id": row_id, "row_ptr": row_ptr, "col_id": col_id}
        # print(mydata)
        # container = torch.jit.script(Container(mydata))
        # torch.jit.save(container, f"sputnik_data.pt")
        # torch.jit.save(torch.jit.script(self.lin), f"sputnik_lin.pt")
        # import os
        # os._exit(0)

        return mygraph.sputnik_gat(x, row_id, row_ptr, col_id, self.lin.weight, \
                                 self.att_i.data, self.att_j.data, self._num_heads, self._out_feats)

class SputnikGAT(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(SputnikGAT, self).__init__()

        self.conv1 = SputnikGATLayer(in_dim, hidden_dim, 1)
        self.conv2 = SputnikGATLayer(hidden_dim, out_dim, 1)

    def forward(self, x, adj, row_ptr):
        h = self.conv1(x, adj[1], row_ptr, adj[0])
        h = F.elu(h)
        h = self.conv2(h, adj[1], row_ptr, adj[0])
        return h