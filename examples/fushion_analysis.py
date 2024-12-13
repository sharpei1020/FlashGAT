from graphiler.utils import load_data, homo_dataset, setup
import sys
import mygraph
import torch
import numpy as np
from scipy.sparse import coo_matrix
import TCGNN

# L1 -> shared | register -> shared | shared -> register (whole)
# global -> register | register -> global (whole)
# L1 -> L2 | L2 -> L1
# dev_mem -> 
DIM = 32
device = setup()

def analysis_agnn_fushion(dataset, feat_dim):
    g, feature = load_data(dataset, feat_dim, prepare=False)
    num_node = feature.shape[0]
    u, v = g.edges()
    adj = torch.vstack([u, v]).type(torch.IntTensor).to(device)
    dev_idx = torch.arange(0, num_node).type(torch.IntTensor).to(device)
    RowWindowOffset, TCOffset, BlockMask, SparseAToX = mygraph.process_DTC(adj, dev_idx, 16, 8, num_node)

    block = (num_node + 15) // 16
    BLK_num = RowWindowOffset[-1].item()
    ########## Analysis ##########
    # L1 -> shared (x_i->D | x_j->dense_X| )
    total_shared_read = (block * 16 * feat_dim + BLK_num * 8 * (feat_dim + 1)) * 4 / 1024 / 1024
    print("Total shared read: {:.2f} MB, nsight compute diff: {:.2f} MB".format(total_shared_read, 177.93 - total_shared_read))
    # shared -> register (D->A | dense_X->frag_B | dense_x->B | D->(reg) | sparse_X | softmax | )
    total_shared2reg = total_shared_read + (block * 16 * feat_dim * 4 + BLK_num * 8 * feat_dim * 4 + BLK_num * 8 * 16 * 9 * 4 + BLK_num * 21 * 16 * 4) / 1024 / 1024
    print("Each_kernel_shared2reg: {:.3f} MB, nsight compute diff: {:.3f} MB".format(total_shared2reg / block * 0.4667 * 280, 15.51 - total_shared2reg / block * 0.4667 * 280))
    # register -> shared (sparse_X(init+atomic_add+) | softmax | D | )
    total_reg2shared = (BLK_num * 16 * 8 * 6 * 4 + BLK_num * 4 * 16 * 4 + block * 16 * feat_dim * 4) / 1024 / 1024
    print("Each_kernel_reg2shared: {:.3f} MB, nsight compute diff: {:.3f} MB".format(total_reg2shared / block * 0.4667 * 280, 7.22 - total_reg2shared / block * 0.4667 * 280))
    # L1 -> register ((e_start, e_end, b_start, b_end, b, block_high, node_len) | block_width | x_norm_i | SparseAToX->rowid | BlockMask->mask | )
    total_register_read = ((8 * 2 * block + 4 * 5 * block + 4 * block) * 4 + 4 * (BLK_num - block) * 4 + 2 * 8 * 4 * block + 8 * 4 * BLK_num + 16 * 1 * BLK_num) / 1024 / 1024
    print("Total register read: {:.2f} MB, nsight compute diff: {:.2f} MB".format(total_register_read / block * 0.4667 * 280, 2.12 - total_register_read / block * 0.4667 * 280))
    # L2 -> L1 (x_i | x_j | x_norm_i | x_norm_j | TCOffset | RowWindowOffset | BlockMask | SparseAToX | )
    total_L1_read = (block * 16 * feat_dim * 4 + BLK_num * 8 * feat_dim * 4 + block * 16 * 4 * 4 + BLK_num * 8 * 4 * 4 + 2 * block * (4 + 8) * 4 + BLK_num * 8 * 16 + BLK_num * 8 * 4 + block * 4 * 4 + (BLK_num - 1) * 4) / 1024 / 1024
    print("Total L1 read: {:.2f} MB, nsight compute diff: {:.2f} MB".format(total_L1_read, 218.66 - total_L1_read))
    # L1 -> L2
    total_L1_write = num_node * feat_dim * 4 / 1024 / 1024
    print("Total L1 write: {:.2f} MB, nsight compute diff: {:.2f} MB".format(total_L1_write, 21.46 - total_L1_write))

def analysis_agnn_no_fushion(dataset, feat_dim):
    g, feature = load_data(dataset, feat_dim, prepare=False)
    u, v = g.edges()
    num_nodes = g.num_nodes()
    num_edges = len(u)
    val = np.array([1] * num_edges, dtype=np.float32)
    edge_index = np.stack((u, v))
    scipy_csr = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes)).tocsr()
    row_ptr = torch.tensor(scipy_csr.indptr)
    col_idx = torch.tensor(scipy_csr.indices)
    num_row_windows = (num_nodes + 16 - 1) // 16
    edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
    edgeToRow = torch.zeros(num_edges, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    TCGNN.preprocess(col_idx, row_ptr, num_nodes,  \
                        16,	8, blockPartition, edgeToColumn, edgeToRow)
    ########## Analysis SDDMM ##########
    #L2 -> L1 (row_ptr | col_idx | blockPartition | edgeToColumn | edgeToRow | param | )
    total_L1_read = (num_nodes * 3 * 4 + num_edges * 4 + 8 * num_edges * feat_dim * 4 + \
                    num_edges * 4 * 18 + 8 * num_row_windows * 4 + \
                    4 * ((blockPartition * 8 + 15) // 16).sum().item() * 11 + \
                    19 * 4 * num_row_windows) / 1024 / 1024 / 1024
    print("Total L1 read: {:.2f} GB, nsight compute: {:.2f} GB, diff: {:.2f} GB".format(total_L1_read, 1.28, 1.28 - total_L1_read))
    #L1 -> L2 (feature)
    total_L1_write = num_edges * 4 / 1024 / 1024
    print("Total L1 write: {:.2f} MB, nsight compute: {:.2f} MB, diff: {:.2f} MB".format(total_L1_write, 34.75, 34.75 - total_L1_write))
    ########## Analysis SpMM ##########
    #L2 -> L1 (row_ptr | col_idx | blockPartition | edgeToColumn | edgeToRow | )
    total_L1_read = (num_nodes * 3 * 4 + num_edges * 4 * 4 + blockPartition.sum() * 8 * feat_dim * 4 * 4) / 1024 / 1024
    print("Total L1 read: {:.2f} MB, nsight compute: {:.2f} MB, diff: {:.2f} MB".format(total_L1_read, 651.68, 651.68 - total_L1_read))
    #L1 -> L2 (feature)
    total_L1_write = num_nodes * feat_dim * 4 * 4 / 1024 / 1024
    print("Total L1 write: {:.2f} MB, nsight compute: {:.2f} MB, diff: {:.2f} MB".format(total_L1_write, 86.70, 86.70 - total_L1_write))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fushion_analysis.py [dataset]")
        exit()
    if sys.argv[1] == "all":
        for d in homo_dataset:
            # analysis_agnn_fushion(d, DIM)
            analysis_agnn_no_fushion(d, DIM)
    else:
        # analysis_agnn_fushion(sys.argv[1], DIM)
        analysis_agnn_no_fushion(sys.argv[1], DIM)