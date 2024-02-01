from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_sparse import SparseTensor
import torch
from torch_geometric.typing import torch_sparse

if __name__ == '__main__':
    src = torch.tensor([0, 1, 2])
    dist = torch.tensor([1, 2, 0])
    adj = SparseTensor(row=src, col=dist, sparse_sizes=(3, 3)).to("cuda:0")
    adj.fill_value(1.)
    adj = torch_sparse.fill_diag(adj, 1.)
    print(adj.to_dense())
    deg = torch_sparse.sum(adj, dim=1)
    print(deg)