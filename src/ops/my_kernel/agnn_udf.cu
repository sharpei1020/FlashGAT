#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/sort.h>
#include "agnn.cuh"

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define BLK_H 16 
#define BLK_W 8
#define FULL_MASK 0xffffffff

//softmax(head=1)
__global__ void softmax_v(const float *__restrict__ features, const int *__restrict__ pointer,
                          const int *__restrict__ indices, float *__restrict__ next_layer) {
  int neighbor_offset = pointer[blockIdx.x];
  int degree = pointer[blockIdx.x + 1] - neighbor_offset;

  float max_local = 0.0f;
  for (int i = 0; i < degree / 32; i++) {
    max_local = max(features[indices[neighbor_offset + i * 32 + threadIdx.x]],
                    max_local);
  }
  if (threadIdx.x < degree % 32) {
    max_local = max(features[indices[neighbor_offset + degree - (degree % 32) +
                                     threadIdx.x]],
                    max_local);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    max_local = max(__shfl_down_sync(FULL_MASK, max_local, offset), max_local);
  }
  max_local = __shfl_sync(FULL_MASK, max_local, 0);

  float exp_local = 0.0f;
  for (int i = 0; i < degree / 32; i++) {
    exp_local += expf(
        features[indices[neighbor_offset + i * 32 + threadIdx.x]] - max_local);
  }
  if (threadIdx.x < degree % 32) {
    exp_local += expf(features[indices[neighbor_offset + degree -
                                       (degree % 32) + threadIdx.x]] -
                      max_local);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    exp_local += __shfl_down_sync(FULL_MASK, exp_local, offset);
  }
  float sum_exp_local = 1 / __shfl_sync(FULL_MASK, exp_local, 0);

  for (int i = 0; i < degree / 32; i++) {
    int neighbor = indices[neighbor_offset + i * 32 + threadIdx.x];
    next_layer[neighbor] = expf(features[neighbor] - max_local) * sum_exp_local;
  }
  if (threadIdx.x < degree % 32) {
    int neighbor =
        indices[neighbor_offset + degree - (degree % 32) + threadIdx.x];
    next_layer[neighbor] = expf(features[neighbor] - max_local) * sum_exp_local;
  }
  return;
}

__global__ void reorder_csr_metcf(const float *next_layer, const int* __restrict__ pointer,
                            const int* __restrict__ edgeToColumn, const int* __restrict__ RowWindow_offset,
                            float *__restrict__ features, int numNodes) {
    int element_start = pointer[blockIdx.x*16];
    int element_end = pointer[min((blockIdx.x+1)*16, numNodes)];
    int block_start = RowWindow_offset[blockIdx.x];
    int block_end = RowWindow_offset[blockIdx.x+1];
    __shared__ uint32_t mask[8];
    __shared__ int offset[1];
    if (threadIdx.x == 0)
        offset[0] = 0;
    for (int i = block_start; i < block_end; i++) {
        if ((threadIdx.x & 31) == 0)
            mask[threadIdx.x >> 5] = 0;
        for (int j = element_start + threadIdx.x; j < element_end; j += blockDim.x) {
            int col = edgeToColumn[j];
            int set = ((col >= 8 * (i - block_start)) && (col < (8 * (i - block_start) + 8))) << (threadIdx.x & 31);
            for (int k = 1; k < 32; k <<= 1)
                set |= __shfl_xor_sync(FULL_MASK, set, 2*k-1);
            mask[threadIdx.x >> 5] = set;
            __syncthreads();
            if (set != 0) {
                int off = __popc(set & ((1<<threadIdx.x&31)-1));
                for (int l = 0; l < (threadIdx.x >> 5); l++)
                    off += __popc(mask[l]);
                features[element_start + offset[0] + off] = next_layer[j];
            }
            __syncthreads();
            if (threadIdx.x == 0)
                for (int k = 0; k < 4; k++)
                    offset[0] += __popc(mask[k]);
        }
    }
}

//DTC-SpMM
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}

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
////////////////////////////////////////////
    int tag
) {
    int num_nodes = feature.size(0);
    int num_edges = edgeToColumn.size(0);
    if (tag == 1) {
        auto next_layer = torch::empty({1, num_edges}, attention_feat.options());
        softmax_v<<<num_nodes, 32>>>(
            attention_feat.data_ptr<float>(),
            row_pointers.data_ptr<int>(),
            column_index.data_ptr<int>(),
            next_layer.data_ptr<float>()
        );
        int block = (num_nodes + 255) / 256;
        int thread = 256;

        reorder_csr_metcf<<<block, thread>>>(
            next_layer.data_ptr<float>(),
            row_pointers.data_ptr<int>(),
            edgeToColumn.data_ptr<int>(),
            Rowwindow_offset.data_ptr<int>(),
            attention_feat.data_ptr<float>(),
            num_nodes);
        return feature;
    } else if (tag == 2) {
        int block = (num_nodes + 15) / 16;
        int thread = 128;

        auto output = torch::empty({num_nodes, 32}, feature.options());

        spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split<<<block, thread>>>(
            Rowwindow_offset.data_ptr<int>(), TCblocktile_id.data_ptr<uint8_t>(), TCblock_offset.data_ptr<int>(),
            sparseAToXidx.data_ptr<int>(), attention_feat.data_ptr<float>(), num_nodes, num_edges, 32, 
            feature.data_ptr<float>(), output.data_ptr<float>());
        return output;
    } else {
        auto next_layer = torch::empty({1, num_edges}, attention_feat.options());
        softmax_v<<<num_nodes, 32>>>(
            attention_feat.data_ptr<float>(),
            row_pointers.data_ptr<int>(),
            column_index.data_ptr<int>(),
            next_layer.data_ptr<float>()
        );
        int block = (num_nodes + 255) / 256;
        int thread = 256;

        reorder_csr_metcf<<<block, thread>>>(
            next_layer.data_ptr<float>(),
            row_pointers.data_ptr<int>(),
            edgeToColumn.data_ptr<int>(),
            Rowwindow_offset.data_ptr<int>(),
            attention_feat.data_ptr<float>(),
            num_nodes);

        block = (num_nodes + 15) / 16;
        thread = 128;

        auto output = torch::empty({num_nodes, 32}, feature.options());

        spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split<<<block, thread>>>(
            Rowwindow_offset.data_ptr<int>(), TCblocktile_id.data_ptr<uint8_t>(), TCblock_offset.data_ptr<int>(),
            sparseAToXidx.data_ptr<int>(), attention_feat.data_ptr<float>(), num_nodes, num_edges, 32, 
            feature.data_ptr<float>(), output.data_ptr<float>());

        return output;
    }

    // auto next_layer = torch::empty({1, num_edges}, attention_feat.options());
    // softmax_v<<<num_nodes, 32>>>(
    //     attention_feat.data_ptr<float>(),
    //     row_pointers.data_ptr<int>(),
    //     column_index.data_ptr<int>(),
    //     next_layer.data_ptr<float>()
    // );
    // int block = (num_nodes + 255) / 256;
    // int thread = 256;

    // reorder_csr_metcf<<<block, thread>>>(
    //     next_layer.data_ptr<float>(),
    //     row_pointers.data_ptr<int>(),
    //     edgeToColumn.data_ptr<int>(),
    //     Rowwindow_offset.data_ptr<int>(),
    //     attention_feat.data_ptr<float>(),
    //     num_nodes);

    // block = (num_nodes + 15) / 16;
    // thread = 128;

    // auto output = torch::empty({num_nodes, 32}, feature.options());

    // spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split<<<block, thread>>>(
    //     Rowwindow_offset.data_ptr<int>(), TCblocktile_id.data_ptr<uint8_t>(), TCblock_offset.data_ptr<int>(),
    //     sparseAToXidx.data_ptr<int>(), attention_feat.data_ptr<float>(), num_nodes, num_edges, 32, 
    //     feature.data_ptr<float>(), output.data_ptr<float>());

    // return output;
}  

__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
        __shared__ unsigned offset[1]; 
        __shared__ unsigned mask[8];
        int tid = threadIdx.x;
        int winId = blockIdx.x; // each warp one window
        unsigned block_start = rowwindow_offset[winId];
        unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
        unsigned num_blocks = block_end - block_start;
        if (num_blocks == 0) 
            return;
        int *tcblock_offset_global_ptr = tcblock_offset + block_start;
        unsigned element_start = nodePointer[winId * blockSize_h];
        unsigned element_end =
            nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
        if (threadIdx.x == 0)
            offset[0] = 0;
        __syncthreads();
        auto tileid = tcblocktile_id + element_start;
        for (int i = 0; i < num_blocks; i++) {
            for (unsigned e_index = element_start + tid; e_index < element_end; e_index += blockDim.x) {
                unsigned col = edgeToColumn[e_index]; // new col
                if (i == 0)
                    atomicAdd(tcblock_offset_global_ptr + col / blockSize_w, 1);
                if ((threadIdx.x&31)==0)
                    mask[threadIdx.x>>5]=0;
                int set = ((col >= blockSize_w * i && col < blockSize_w * (i + 1))<<(threadIdx.x&31));
                for (int j = 1; j < 32; j <<= 1)
                    set |= __shfl_xor_sync(0xffffffff, set, j*2-1);
                mask[threadIdx.x>>5] = set;
                __syncthreads();
                if (col >= blockSize_w * i && col < blockSize_w * (i + 1)) {
                    unsigned row_local = edgeToRow[e_index] % blockSize_h;
                    unsigned col_local = col % blockSize_w;
                    unsigned off = __popc(set & ((1<<(threadIdx.x&31))-1));
                    for (int j = 0; j < (threadIdx.x>>5); j++)
                        off += __popc(mask[j]);
                    tileid[offset[0]+off] = (uint8_t)(row_local * blockSize_w + col_local);
                    sparseatob[(block_start + i) * blockSize_w + col_local] = edgeList[e_index];
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    for (int j = 0; j < 4; j++)
                        offset[0] += __popc(mask[j]);
            }
        }      
}

std::vector<at::Tensor> DTC_compression(
    at::Tensor row_pointers,
    at::Tensor column_index,
    at::Tensor blockPartition,
    at::Tensor edgeToColumn,
    at::Tensor edgeToRow
) {
    int num_nodes = row_pointers.size(0) - 1;
    int num_edges = edgeToColumn.size(0);

    auto RowWindow_offset = torch::empty({blockPartition.size(0) + 1}, blockPartition.options().dtype(torch::kInt));
    auto TCblocktile_id = torch::empty({num_edges}, blockPartition.options().dtype(torch::kUInt8));
    cudaMemset(RowWindow_offset.data_ptr<int>(), 0, sizeof(int));
    thrust::inclusive_scan(thrust::device, blockPartition.data_ptr<int>(), blockPartition.data_ptr<int>() + blockPartition.size(0), 
        RowWindow_offset.data_ptr<int>() + 1);
    int block_num;
    cudaMemcpy(&block_num, RowWindow_offset.data_ptr<int>() + blockPartition.size(0), sizeof(int), cudaMemcpyDeviceToHost);
    auto TCblock_offset = torch::empty({block_num+1}, blockPartition.options().dtype(torch::kInt));
    auto sparseAToXidx = torch::empty({block_num*8}, blockPartition.options().dtype(torch::kInt));
    thrust::fill_n(thrust::device, TCblock_offset.data_ptr<int>(), block_num+1, 0);
    thrust::fill_n(thrust::device, sparseAToXidx.data_ptr<int>(), block_num*8, num_nodes);
    int block = (num_nodes + 15) / 16;
    int thread = 256;
    generate_tcoffset_id_atob<<<block, thread>>>(
        row_pointers.data_ptr<int>(), RowWindow_offset.data_ptr<int>(), edgeToColumn.data_ptr<int>(), edgeToRow.data_ptr<int>(),
        column_index.data_ptr<int>(), TCblock_offset.data_ptr<int>(), TCblocktile_id.data_ptr<uint8_t>(), 
        sparseAToXidx.data_ptr<int>(), num_nodes, 16, 8, block);
    thrust::inclusive_scan(thrust::device, TCblock_offset.data_ptr<int>() + 1, TCblock_offset.data_ptr<int>() + block_num+1, 
        TCblock_offset.data_ptr<int>() + 1);
    return {RowWindow_offset, TCblock_offset, TCblocktile_id, sparseAToXidx};
}