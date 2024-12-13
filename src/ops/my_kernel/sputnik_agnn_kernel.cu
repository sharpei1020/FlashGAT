#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include "../../sputnik/sputnik.h"


at::Tensor sputnik_AGNN(
    at::Tensor feature,
    at::Tensor row_idx,
    at::Tensor row_offset,
    at::Tensor col_idx,
    at::Tensor beta,
    int out_feats
) {
    auto x_norm = torch::nn::functional::normalize(feature);
    auto x_norm_t = x_norm.transpose(0, 1).contiguous();
    int num_edges = row_idx.size(0);
    int num_nodes = feature.size(0);
    int num_feats = feature.size(1);
    float* alpha, *alpha_softmax;
    cudaMalloc(&alpha, sizeof(float) * num_edges);
    sputnik::CudaSddmm(num_nodes, num_feats, num_nodes, num_edges, row_idx.data_ptr<int>(), 
        row_offset.data_ptr<int>(), col_idx.data_ptr<int>(), x_norm.data_ptr<float>(), 
        x_norm_t.data_ptr<float>(), alpha, c10::cuda::getCurrentCUDAStream());
    float b = beta[0].item<float>();
    thrust::transform(thrust::device, alpha, alpha + num_edges, alpha, [=]__device__(float x) {return x * b;});
    cudaMalloc(&alpha_softmax, sizeof(float) * num_edges);
    sputnik::SparseSoftmax(num_nodes, num_nodes, num_edges, 
                    alpha, row_idx.data_ptr<int>(), row_offset.data_ptr<int>(), 
                    col_idx.data_ptr<int>(), alpha_softmax, 
                    c10::cuda::getCurrentCUDAStream());
    cudaFree(alpha);
    auto output = torch::empty({num_nodes, num_feats}, feature.options());
    sputnik::CudaSpmm(num_nodes, num_nodes, num_feats, num_edges,
                    row_idx.data_ptr<int>(), alpha_softmax, 
                    row_offset.data_ptr<int>(), col_idx.data_ptr<int>(), feature.data_ptr<float>(), 
                    output.data_ptr<float>(), c10::cuda::getCurrentCUDAStream());
    cudaFree(alpha_softmax);
    return output;
}