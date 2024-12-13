#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include "../../sputnik/sputnik.h"
// #include "../../sputnik/test_utils.h"
#include "gat.cuh"


at::Tensor sputnik_GAT(
    at::Tensor feature,
    at::Tensor row_idx,
    at::Tensor row_offset,
    at::Tensor col_idx,
    at::Tensor lin_weight,
    at::Tensor att_i,
    at::Tensor att_j,
    int num_heads,
    int out_feats
) {
    int num_nodes = feature.size(0);
    auto feats = torch::mm(feature, lin_weight.t());
    auto alpha_i = torch::matmul(feats, att_i.squeeze());
    auto alpha_j = torch::matmul(feats, att_j.squeeze());
    int num_edges = row_idx.size(0);
    float *a_i, *a_j, *alpha;
    cudaMalloc(&a_i, num_edges * sizeof(float));
    cudaMalloc(&a_j, num_edges * sizeof(float));
    cudaMalloc(&alpha, num_edges * sizeof(float));
    thrust::gather(thrust::device, col_idx.data_ptr<int>(), col_idx.data_ptr<int>() + num_edges,
                alpha_i.data_ptr<float>(), a_i);
    thrust::gather(thrust::device, row_idx.data_ptr<int>(), row_idx.data_ptr<int>() + num_edges,
                alpha_j.data_ptr<float>(), a_j);
    thrust::transform(thrust::device, a_i, a_i + num_edges, a_j, alpha, thrust::plus<float>());
    cudaFree(a_i);
    cudaFree(a_j);
    auto alpha_softmax = torch::empty({num_edges}, feature.options());
    sputnik::SparseSoftmax(num_nodes, num_nodes, num_edges, 
                    alpha, row_idx.data_ptr<int>(), row_offset.data_ptr<int>(), 
                    col_idx.data_ptr<int>(), alpha_softmax.data_ptr<float>(), 
                    c10::cuda::getCurrentCUDAStream());
    auto output = torch::empty({num_nodes, num_heads * out_feats}, feature.options());
    sputnik::CudaSpmm(num_nodes, num_nodes, num_heads * out_feats, num_edges,
                    row_idx.data_ptr<int>(), alpha_softmax.data_ptr<float>(), 
                    row_offset.data_ptr<int>(), col_idx.data_ptr<int>(), feats.data_ptr<float>(), 
                    output.data_ptr<float>(), c10::cuda::getCurrentCUDAStream());
    return output;   
}