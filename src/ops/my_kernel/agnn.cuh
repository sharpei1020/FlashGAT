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

at::Tensor AGNN_short(
    at::Tensor feature,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta,
    int out_feats,
    int block_high,
    int block_width
);

at::Tensor sputnik_AGNN(
    at::Tensor feature,
    at::Tensor row_idx,
    at::Tensor row_offset,
    at::Tensor col_idx,
    at::Tensor beta,
    int out_feats
);

at::Tensor AGNN_UDF(
    at::Tensor feature,
    at::Tensor attention_feat,
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow,
    at::Tensor Rowwindow_offset,
    at::Tensor TCblocktile_id,
    at::Tensor TCblock_offset,
    at::Tensor sparseAToXidx,
    int tag
);

std::vector<at::Tensor> DTC_compression(
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor blockPartition,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow
);

at::Tensor SDDMM(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta
);

at::Tensor SDDMM_TCGNN(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor blockPartition,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow,
    at::Tensor beta
);

at::Tensor AGNN_divide(
    at::Tensor feature,
    at::Tensor row_pointers,
    at::Tensor RowWindowOffsets,
    at::Tensor SparseAToX,
    at::Tensor BitMaskRowOffset,
    at::Tensor BitColMask,
    at::Tensor BitRowMask,
    at::Tensor beta
);