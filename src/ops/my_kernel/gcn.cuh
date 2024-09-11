#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

// void linear(const float* lin_weight, const float* feature, float* out_feature,
//             int rows, int cols, int weight_rows, int weight_cols);

__global__ void linear(const float* __restrict__ weight, const float* __restrict__ bias, const float* __restrict__ X, float*  __restrict__ Z, int M, int K, bool bias_flag);


// void gather(
//     float* in_f, 
//     float* out_f, 
//     const int* n_out_edge_index,
//     const int* n_out_offsets, 
//     const int* n_out_counts,
//     const int* coo_dst,
//     int rows,
//     int weight_cols,
//     int* thread_map
// );

at::Tensor GCN(
    torch::Tensor lin_weight, 
    torch::Tensor lin_bias,
    torch::Tensor edge_index_0,
    torch::Tensor edge_index_1,
    torch::Tensor count,
    torch::Tensor out,
    torch::Tensor x,
    int out_feats);