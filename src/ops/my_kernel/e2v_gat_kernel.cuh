#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

at::Tensor E2V_GAT(
    at::Tensor x,
    at::Tensor k_feature,
    at::Tensor v_feature,
    at::Tensor edge_id,
    at::Tensor row_offset,
    int num_head,
    int block_size
);