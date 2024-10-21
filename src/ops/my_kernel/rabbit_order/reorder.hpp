#include <torch/extension.h>

std::vector<torch::Tensor> rabbit_reorder(
    torch::Tensor in_edge_index
);
