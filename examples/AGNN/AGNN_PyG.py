import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv

class AGNN_PyG(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dim):
        super(AGNN_PyG, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            self.convs.append(AGNNConv(requires_grad=False))
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.lin2(x)
        return x
        