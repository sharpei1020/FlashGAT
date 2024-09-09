import torch
from torch import Tensor
import torch.nn as nn
from typing import Any, Optional, List
import math
import copy
from torch_geometric.utils import scatter, softmax, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
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

def zeros(value: Any):
    constant(value, 0.)

def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None,
             deg_test: Optional[Tensor] = None):
    fill_value = 2. if improved else 1.
    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
################################################################
    if deg_test is not None:
        condition_deg = torch.all(torch.abs(deg_inv_sqrt - deg_test) < 1e-5, dim=-1).cpu()
        print(torch.nonzero(torch.where(condition_deg, torch.zeros(condition_deg.shape), torch.ones(condition_deg.shape))))
        assert(torch.all(condition_deg).item())
#####################################################################
    return edge_index, edge_weight

class SumAggregation(nn.Module):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, dim_size, dim, reduce='sum')
     
    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               dim_size: Optional[int] = None, dim: int = -2, reduce: str = 'sum') -> Tensor:

        assert index is not None
        return scatter(x, index.type(torch.int64), dim, dim_size, reduce)

class MyGCNlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MyGCNlayer, self).__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_dim))
        self.aggr_module = SumAggregation()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias.data)

    def forward(self, x, edge_index, counts, out_edge_index, layer_i):
############################################################
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": x, "edge_index_0": edge_index[0], "edge_index_1": edge_index[1], 
        #           "counts": counts, "out_edge_index": out_edge_index}
        
        # container = torch.jit.script(Container(mydata))
        # torch.jit.save(container, f"{layer_i}_data.pt")

        # torch.jit.save(torch.jit.script(self.lin), f"{layer_i}_lin.pt")
        # torch.jit.save(torch.jit.script(Container({"att_i": self.att_i.data})), f"{layer_i}_att_i.pt")
        # torch.jit.save(torch.jit.script(Container({"att_j": self.att_j.data}), f"{layer_i}_att_j.pt"))
        # x_test = torch.zeros((x.size(0), self._out_feats * self._num_heads)).to("cuda")
        # deg_test = torch.ones(x.size(0)).to("cuda")
##################################################################

        bias = torch.zeros((self.lin.weight.shape[0],)).to("cuda")
        # print(edge_index[0], edge_index[1], counts)
        out = mygraph.gcn(self.lin.weight, bias, edge_index[0], edge_index[1], counts, out_edge_index, x, 64)
        # edge_weight = None
        # edge_index, edge_weight = gcn_norm(  # yapf: disable
        #                 edge_index, edge_weight, x.size(-2),
        #                 False, True, 'source_to_target', x.dtype)     
        # x = self.lin(x)
        # print(edge_index[0], edge_index[1])
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight)     
        # out += self.bias.data
#######################################################
        # print(out, out_test)
        # print(out.shape, out_test.shape)
        # condition_out = torch.all(torch.abs(out - out_test) < 1e-5, dim=-1).cpu()
        # print(torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape))))
        # assert(torch.all(condition_out).item())
#########################################################
        return out
    
    def message(self, x_j: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def propagate(self, edge_index: Tensor, x: Tensor, edge_weight: Optional[Tensor] = None):
        size = [None, None]
        data = self._collect(['x_j'], edge_index, size, x)
        out = self.message(data['x_j'], edge_weight)
        out = self.aggr_module(out, data['index'], dim_size=data['dim_size'])
        return out

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
        out['ptr'] = None
        out['size'] = size
        out['index'] = edge_index[i]
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']
        return out
    
    def _set_size(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(-2)
        elif the_size != src.size(-2):
            raise ValueError(
                (f'Encountered tensor with size {src.size(-2)} in '
                 f'dimension {-2}, but expected size {the_size}.'))
        
    def _lift(self, src, edge_index, dim):
        index = edge_index[dim]
        return src.index_select(-2, index)
    

class MyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MyGCN, self).__init__()

        self.layer1 = MyGCNlayer(in_dim, hidden_dim)
        self.layer2 = MyGCNlayer(out_dim, hidden_dim)

    def forward(self, x, adj, counts, out_edge_index):
        x = self.layer1(x, adj, counts, out_edge_index, 1)
        x = F.relu(x)
        x = self.layer2(x, adj, counts, out_edge_index, 2)
        return x