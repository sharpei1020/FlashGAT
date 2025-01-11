#include <cuda_runtime.h>
#include <cstdint>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define BLK_H 16 
#define BLK_W 8



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

