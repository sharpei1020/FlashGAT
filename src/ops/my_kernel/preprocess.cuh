#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> preprocess_CSR(at::Tensor edge_index, at::Tensor counts, int group, int num_nodes);

std::vector<at::Tensor> process_CSR(at::Tensor edge_index, int group, int num_nodes);

std::vector<at::Tensor> get_graph_set(at::Tensor edge_index, int numwarps, int numnodes);

std::vector<torch::Tensor> process_DTC(at::Tensor edge_index, at::Tensor dev_idx, int block_high, int block_width, int num_nodes, bool balance);

std::vector<at::Tensor> process_DTC_short_mask(at::Tensor edge_index, int block_high, int block_width, int num_nodes, bool balance);