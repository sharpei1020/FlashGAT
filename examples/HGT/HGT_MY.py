import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import List, Optional
from graphiler.utils import setup
from torch_geometric.utils import softmax, scatter
from inspect import Parameter

device = setup()

def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

class SumAggregation(nn.Module):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, dim_size, dim, reduce='sum')
     
    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               dim_size: Optional[int] = None, dim: int = -2, reduce: str = 'sum') -> Tensor:

        assert index is not None
        return scatter(x, index.type(torch.int64), dim, dim_size, reduce)


class  MyHGTLayerSlice(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, num_node_type, num_rels):
        super(MyHGTLayerSlice, self).__init__()
        self.aggr_module = SumAggregation()
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_node_types = num_node_type
        self.num_rels = num_rels

        self.k_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.q_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.v_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.a_weights = torch.rand(
            self.num_node_types, self.out_feat_dim, self.out_feat_dim).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)

        self.skip = torch.ones(self.num_node_types).to(device)

    def upd(self, h, node_type):
        node_type = node_type.squeeze(-1)
        # (N, 1)
        skip = self.skip[node_type]
        # (N, out_dim, out_dim)
        a_weight = self.a_weights[node_type]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(h.unsqueeze(1), a_weight).squeeze()
        return trans_out * alpha.unsqueeze(1)
    
    def forward(self, h, adj, edge_type, node_type, src_type, dst_type):
        out = 0
        k_whole = (h.unsqueeze(1) @ self.k_weights[node_type]).squeeze(1)
        v_whole = (h.unsqueeze(1) @ self.v_weights[node_type]).squeeze(1)
        q_whole = (h.unsqueeze(1) @ self.q_weights[node_type]).squeeze(1)
        data = self._collect(['k_j', 'v_j', 'q_i'], edge_index = adj, 
                            size=[None, None], k=k_whole, v=v_whole, q=q_whole)
        print(data)
        out_ = self.message2(data['k_j'], data['v_j'], data['q_i'], 
                           data['index'], data['size_i'], edge_type)
        print(out_)
        out_ = self.aggr_module(out_, data['index'], dim_size=data['dim_size'])
        # for i in range(self.num_rels):
        #     tmp = masked_edge_index(adj, edge_type == i)
        #     src_ntype = src_type[i]
        #     dst_ntype = dst_type[i]

        #     k = h @ self.k_weights[src_ntype]
        #     v = h @ self.v_weights[src_ntype]
        #     q = h @ self.q_weights[dst_ntype]
        #     k = k @ self.relation_att[i]
        #     v = v @ self.relation_msg[i]
        #     out_i = self.propagate(
        #         tmp, k=k, v=v, q=q, rel_pri=self.relation_pri[i])
        #     out = out + out_i
        print(out_)
        assert torch.equal(out_, out)

        out = self.upd(out, node_type)
        return out
    
    def propagate(self, edge_index, k, v, q, rel_pri):
        size = [None, None]
        data = self._collect(['k_j', 'v_j', 'q_i'], edge_index = edge_index, 
                            size=size, k=k, v=v, q=q)
        out = self.message(data['k_j'], data['v_j'], data['q_i'], 
                           data['index'], data['size_i'], rel_pri)
        out = self.aggr_module(out, data['index'], dim_size=data['dim_size'])
        return out

    def _collect(self, args, edge_index, size, **kwargs):
        i,j = (1, 0) 
        out = {}
        for arg in args:
            data = kwargs.get(arg[:-2], Parameter.empty)
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
    
    def message(self, k_j, v_j, q_i, edge_index_i, size_i, rel_pri):
        t = k_j * q_i
        attn_score = torch.sum(t, dim=1, keepdim=True) * rel_pri / self.sqrt_dk
        alpha = softmax(attn_score, edge_index_i, num_nodes=size_i)
        return v_j * alpha
    
    def message2(self, k_j, v_j, q_i, edge_index_i, size_i, edge_type):
        t = (k_j.unsqueeze(1) @ self.relation_att[edge_type]) * q_i
        attn_score = torch.sum(t, dim=1, keepdim=True) * self.relation_pri[edge_type] / self.sqrt_dk
        index = edge_index_i * self.num_rels + edge_type
        alpha = softmax(attn_score, index, num_nodes=size_i * self.num_rels)
        return (v_j @ self.relation_msg[edge_type]) * alpha
    

class MyHGTLayer(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, num_node_type, num_rels):
        super(MyHGTLayer, self).__init__()
        self.aggr_module = SumAggregation()
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_node_types = num_node_type
        self.num_rels = num_rels

        self.k_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.q_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.v_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.a_weights = torch.rand(
            self.num_node_types, self.out_feat_dim, self.out_feat_dim).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)

        self.skip = torch.ones(self.num_node_types).to(device)

    def upd(self, h, node_type):
        node_type = node_type.squeeze(-1)
        # (N, 1)
        skip = self.skip[node_type]
        # (N, out_dim, out_dim)
        a_weight = self.a_weights[node_type]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(h.unsqueeze(1), a_weight).squeeze()
        return trans_out * alpha.unsqueeze(1)

    def forward(self, h, adj, edge_type, node_type, src_type, dst_type):
        node_type = node_type.unsqueeze(-1)
        h = self.propagate(adj, x=h, edge_type=edge_type, node_type=node_type)
        out = self.upd(h, node_type)
        return out
    
    def propagate(self, edge_index, x, edge_type, node_type):
        size = [None, None]
        data = self._collect(['x_i', 'x_j', 'node_type_i', 'node_type_j'], 
                             edge_index=edge_index, size=size, x=x, node_type=node_type)
        
        out = self.message(data['x_i'], data['x_j'], data['index'], edge_type, 
                           data['node_type_i'], data['node_type_j'], data['size_i'])
        out = self.aggr_module(out, data['index'], dim_size=data['dim_size'])
        return out

    def _collect(self, args, edge_index, size, **kwargs):
        i,j = (1, 0) 
        out = {}
        for arg in args:
            data = kwargs.get(arg[:-2], Parameter.empty)
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

    def message(self, x_i, x_j, edge_index_i, edge_type, node_type_i, node_type_j, size_i):
        node_type_i = node_type_i.squeeze(-1)
        node_type_j = node_type_j.squeeze(-1)
        k_weight = self.k_weights[node_type_j]
        v_weight = self.v_weights[node_type_j]
        q_weight = self.q_weights[node_type_i]

        k = torch.bmm(x_j.unsqueeze(1), k_weight).squeeze()
        v = torch.bmm(x_j.unsqueeze(1), v_weight).squeeze()
        q = torch.bmm(x_i.unsqueeze(1), q_weight).squeeze()

        relation_att = self.relation_att[edge_type]
        relation_msg = self.relation_msg[edge_type]
        relation_pri = self.relation_pri[edge_type]

        k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
        v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
        t = k * q
        attn_score = torch.sum(t, dim=1, keepdim=True) * \
            relation_pri / self.sqrt_dk
        alpha = softmax(attn_score, edge_index_i, num_nodes=size_i)

        rst = v * alpha
        return rst
    
class MyHGT(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_node_types, num_rels, mode='bmm'):
        super(MyHGT, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_node_types = num_node_types
        self.num_rels = num_rels

        Layer = MyHGTLayer if mode == 'bmm' else MyHGTLayerSlice

        self.layer0 = Layer(
            self.in_dim, self.h_dim, self.num_node_types, self.num_rels)
        self.layer1 = Layer(
            self.h_dim, self.out_dim, self.num_node_types, self.num_rels)

    def forward(self, adj, h, edge_type, node_type, src_type, dst_type):
        h = self.layer0(h, adj, edge_type, node_type, src_type, dst_type)
        h = self.layer1(h, adj, edge_type, node_type, src_type, dst_type)
        return h

