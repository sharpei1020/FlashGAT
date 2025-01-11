import torch 
import torch.nn as nn 
import mygraph

class UDFAGNNlayer(nn.Module):
    def __init__(self, 
                 feat_dim, requires_grad=True):
        super(UDFAGNNlayer, self).__init__()

        self.feat_dim = feat_dim
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.beta = nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('beta', torch.ones(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.fill_(1.0)

    def forward(self, x, 