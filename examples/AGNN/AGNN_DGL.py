import torch.nn as nn
from dgl.nn.pytorch import AGNNConv

class AGNN_DGL(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_layers,
                 init_beta=1,
                 learn_beta=1):
        super(AGNN_DGL, self).__init__()
        self.g = g
                
        self.proj = nn.Sequential(
            nn.Linear(in_feats, n_hidden),
            nn.ReLU()
        )

        self.layers = nn.ModuleList(
            [AGNNConv(init_beta, learn_beta, allow_zero_in_degree=True) for _ in range(n_layers)]
        )

        self.cls = nn.Sequential(
            nn.Linear(n_hidden, n_hidden)
        )

    def forward(self, features):
        h = self.proj(features)
        for layer in self.layers:
            h = layer(self.g, h)
        return self.cls(h)