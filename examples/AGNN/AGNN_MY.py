import torch
import torch.nn as nn
import mygraph
from torch_geometric.nn import AGNNConv
from torch_sparse import SparseTensor


class MyAGNNlayer(nn.Module):
    def __init__(self, 
                 feat_dim, reuires_grad=True):
        super(MyAGNNlayer, self).__init__()
        
        self.require_grad = reuires_grad
        self.feat_dim = feat_dim
        # self.test_conv = AGNNConv(requires_grad=reuires_grad)
        if reuires_grad:
            self.beta = nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.require_grad:
            self.beta.data.fill_(1)

    def forward(self, adj, x, 
            RowWindowOffset, TCOffset,
            BlockMask, SparseAToX):
        # class Container(torch.nn.Module):
        #     def __init__(self, mydata):
        #         super(Container, self).__init__()
        #         for key, value in mydata.items():
        #             self.register_buffer(key, value)

        # mydata = {"x": x, "RowWindowOffset": RowWindowOffset, "TCOffset": TCOffset, 
        #           "BlockMask": BlockMask, "SparseAToX": SparseAToX, "beta": self.beta}
        
        # container = torch.jit.script(Container(mydata))
        # torch.jit.save(container, f"data.pt")
        # import os
        # os._exit(0)
        # adj_ = SparseTensor(row=adj[0], col=adj[1], sparse_sizes=(x.size(0), x.size(0))).cuda()
        # out_ = self.test_conv(x, adj.cuda())
#         out = mygraph.agnn(x, RowWindowOffset, TCOffset, BlockMask, SparseAToX, self.beta, self.feat_dim)
########################################################################
#         condition_out = torch.all(torch.abs(out_ - out) < 1e-1, dim=-1).cpu()
#         idxs = torch.nonzero(torch.where(condition_out, torch.zeros(condition_out.shape), torch.ones(condition_out.shape)))
#         # idxs = torch.nonzero(torch.where(condition_out, torch.ones(condition_out.shape), torch.zeros(condition_out.shape)))
#         print(out_[idxs[0]], out[idxs[0]])
#         print(idxs.shape[0], x.shape[0], idxs[:100])
#         assert(torch.all(condition_out).item())
##########################################################################   
        # import time
        # t = time.time()   
        out = mygraph.agnn(x, RowWindowOffset, TCOffset, BlockMask, SparseAToX, self.beta, self.feat_dim)
        # torch.cuda.synchronize()
        # print("AGNN time:{:.4f} ms".format(1000*(time.time() - t)))
        return out
    

class MyAGNN(nn.Module):
    def __init__(self, 
                 in_dim, hidden_dim, out_dim):
        super(MyAGNN, self).__init__()

        self.lin1 = nn.Linear(in_dim, hidden_dim)  
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            self.convs.append(MyAGNNlayer(hidden_dim, False))
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, adj, x, RowWindowOffset, TCOffset, BlockMask, SparseAToX):
        x = self.relu(self.lin1(x))
        for conv in self.convs:
            x = self.relu(conv(adj, x, RowWindowOffset, TCOffset, BlockMask, SparseAToX))
        x = self.lin2(x)
        return x
    
class SputnikAGNNLayer(nn.Module):
    def __init__(self,
                 feat_dim, reuires_grad=True):
        super(SputnikAGNNLayer, self).__init__()

        self.require_grad = reuires_grad
        self.feat_dim = feat_dim
        # self.test_conv = AGNNConv(requires_grad=reuires_grad)
        if reuires_grad:
            self.beta = nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.require_grad:
            self.beta.data.fill_(1)

    def forward(self, x, row_id, row_ptr, col_id):
        return mygraph.sputnik_agnn(x, row_id, row_ptr, col_id, self.beta, self.feat_dim)
    
class SputnikAGNN(nn.Module):
    def __init__(self,
                 in_dim, hidden_dim, out_dim):
        super(SputnikAGNN, self).__init__()

        self.lin1 = nn.Linear(in_dim, hidden_dim)  
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            self.convs.append(SputnikAGNNLayer(hidden_dim, False))
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, adj, row_ptr):
        x = self.relu(self.lin1(x))
        for conv in self.convs:
            x = self.relu(conv(x, adj[1], row_ptr, adj[0]))
        x = self.lin2(x)
        return x
    
class MyAGNNlayer_new(MyAGNNlayer):
    def __init__(self, 
                 feat_dim, reuires_grad=True):
        super(MyAGNNlayer_new, self).__init__(feat_dim, reuires_grad)
        
    def reset_parameters(self):
        if self.require_grad:
            self.beta.data.fill_(1)

    def forward(self, x, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask):
        return mygraph.agnn_short(
            x, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask, self.beta, self.feat_dim
        )

class MyAGNN_new(nn.Module):
    def __init__(self, 
                 in_dim, hidden_dim, out_dim):
        super(MyAGNN_new, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)  
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            self.convs.append(MyAGNNlayer_new(hidden_dim, False))
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(True)
    
    def forward(self, x, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask):
        x = self.relu(self.lin1(x))
        for conv in self.convs:
            x = self.relu(conv(x, RowWindowOffset, SparseAToX, BitMaskRowOffset, BitColMask, BitRowMask))
        x = self.lin2(x)
        return x