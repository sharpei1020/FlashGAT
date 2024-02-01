#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#include "gcn.cuh"
#include "assert.h"


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
) {
    const float *lin_w = lin_weight.data_ptr<float>();
    const int *coo_src = coosrc.data_ptr<int>();
    const int *coo_dst = coodst.data_ptr<int>();
    const int *n_out_counts = node_out_counts.data_ptr<int>();
    const int *n_out_offsets = node_out_offsets.data_ptr<int>();
    const int *n_out_edge_index = node_out_edge_index.data_ptr<int>();
    const float *f = feature.data_ptr<float>();
    float *out_f = out_feature.data_ptr<float>();
    int *map = thread_map.data_ptr<int>();

    float *mid_f;
    int *map_now;
    cudaMalloc((void**)&mid_f, sizeof(float) * rows * weight_cols);
    cudaMalloc((void**)&map_now, sizeof(int) * rows);

    linear(lin_w, f, mid_f, rows, cols, weight_rows, weight_cols);

    assert(rows == cols);
    gather(
        mid_f, 
        out_f, 
        n_out_edge_index, 
        n_out_offsets, 
        n_out_counts,
        coo_dst, 
        rows, 
        weight_cols,
        map_now);
    
}