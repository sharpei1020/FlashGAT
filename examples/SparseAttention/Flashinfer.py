import torch
import flashinfer

num_qo_heads = 32

num_kv_heads = 8

head_dim = 128

# allocate 128MB workspace buffer

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)

# sparse mask: [[0, 0, 1], [1, 0, 1], [0, 1, 1]]

M = 3

N = 3

indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")

indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")

bsr_wrapper.plan(

    indptr,

    indices,

    M,

    N,

    1, # R(block_rows)=1

    1, # C(block_columns)=1

    num_qo_heads,

    num_kv_heads,

    head_dim,

)

q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")

k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")

v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")



o = bsr_wrapper.run(q, k, v)

# use dense implementation with attention mask for comparison

mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")

o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)

print(torch.allclose(o, o_ref))