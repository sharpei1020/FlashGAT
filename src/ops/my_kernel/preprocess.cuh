#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> preprocess_CSR(at::Tensor edge_index, at::Tensor counts, int group, int num_nodes);

std::vector<at::Tensor> process_CSR(at::Tensor edge_index, int group, int num_nodes);

std::vector<at::Tensor> get_graph_set(at::Tensor edge_index, int numwarps, int numnodes);

std::vector<torch::Tensor> process_DTC(at::Tensor edge_index, at::Tensor dev_idx, int block_high, int block_width, int num_nodes, bool balance);

std::vector<at::Tensor> process_DTC_short_mask(at::Tensor edge_index, int block_high, int block_width, int num_nodes, bool balance);

std::vector<at::Tensor> SGT_short_Mask(at::Tensor row_pointers, at::Tensor column_index, at::Tensor blockPartition, at::Tensor edgeToColumn,
    at::Tensor edgeToRow, int block_high, int block_width);

std::vector<at::Tensor> ASC_e2v(at::Tensor edge_index, int block_size, int node_num);

std::vector<at::Tensor> adaptive_ASC(at::Tensor edge_index, const std::string &model, int node_num_row, int node_num_col);