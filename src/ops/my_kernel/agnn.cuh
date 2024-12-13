#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

at::Tensor AGNN(
    at::Tensor x,
    at::Tensor RowWindowOffset,
    at::Tensor TCOffset,
    at::Tensor BlockMask,
    at::Tensor SparseAToX,
    at::Tensor beta,
    int out_feats
);

at::Tensor sputnik_AGNN(
    at::Tensor feature,
    at::Tensor row_idx,
    at::Tensor row_offset,
    at::Tensor col_idx,
    at::Tensor beta,
    int out_feats
);