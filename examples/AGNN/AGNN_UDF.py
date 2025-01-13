import torch 
import torch.nn as nn 
import torch.nn.functional as F
import TCGNN
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

    def forward(self, x, row_pointers, 
                column_index, blockPartition, 
                edgeToColumn, edgeToRow, 
                RowWindow_offset, TCblocktile_id,
                TCblock_offset, saprseAToXidx, select_params):
        select_id, edge_attentions = select_params[0], select_params[1]
        # print(select_id, edge_attentions)
        if select_id == -1:
            return x
        elif select_id == 0:
            x_prime = F.normalize(x, p=2, dim=1)
            edge_feature = TCGNN.forward_ef(x_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
            edge_attentions = torch.mm(edge_feature.unsqueeze(-1), self.beta.unsqueeze(0)).transpose(0,1).contiguous()
            return x
        elif select_id >= 1 and select_id < 3:
            return mygraph.agnn_udf(x, edge_attentions,
                                    row_pointers, column_index,
                                    edgeToColumn, edgeToRow,
                                    RowWindow_offset, TCblocktile_id,
                                    TCblock_offset, saprseAToXidx, select_id)
        else:
            x_prime = F.normalize(x, p=2, dim=1)
            edge_feature = TCGNN.forward_ef(x_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
            edge_attentions = torch.mm(edge_feature.unsqueeze(-1), self.beta.unsqueeze(0)).transpose(0,1).contiguous()
            return mygraph.agnn_udf(x, edge_attentions,
                                    row_pointers, column_index,
                                    edgeToColumn, edgeToRow,
                                    RowWindow_offset, TCblocktile_id,
                                    TCblock_offset, saprseAToXidx, select_id)
    
class UDFAGNN(nn.Module):
    def __init__(self, 
                 in_dim, hidden_dim, out_dim):
        super(UDFAGNN, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)  
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            self.convs.append(UDFAGNNlayer(hidden_dim, False))
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, 
                RowWindowOffset, TCblocktile_id, TCblock_offset, saprseAToXidx, select_params): 
        x = self.relu(self.lin1(x))
        for conv in self.convs:
            x = self.relu(conv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, 
                RowWindowOffset, TCblocktile_id, TCblock_offset, saprseAToXidx, select_params))
        x = self.lin2(x)
        return x