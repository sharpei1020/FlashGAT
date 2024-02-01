#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

void linear(const float* lin_weight, const float* feature, float* out_feature,
            int rows, int cols, int weight_rows, int weight_cols);

void gather(
    float* in_f, 
    float* out_f, 
    const int* n_out_edge_index,
    const int* n_out_offsets, 
    const int* n_out_counts,
    const int* coo_dst,
    int rows,
    int weight_cols,
    int* thread_map
);

void gcn(
    const at::Tensor& lin_weight, 
    const at::Tensor& coosrc, 
    const at::Tensor& coodst, 
    const at::Tensor& node_out_counts, 
    const at::Tensor& node_out_offsets, 
    const at::Tensor& node_out_edge_index, 
    int rows, 
    int cols, 
    int edges_num, 
    int weight_rows, 
    int weight_cols, 
    const at::Tensor& feature, 
    at::Tensor& out_feature,
    at::Tensor& thread_map
);