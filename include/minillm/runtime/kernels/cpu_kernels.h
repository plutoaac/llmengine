#pragma once

// Naive CPU kernel declarations matching mini_op function signatures.
// Current implementations are scalar (no SIMD); they will be replaced
// by mini_op SIMD kernels in a future phase.

#include <cstddef>
#include <cstdint>

namespace minillm::cpu {

// GEMM: C[M,N] = A[M,K] @ B[K,N], row-major
void sgemm(const float* A, const float* B, float* C, int M, int N, int K);

// GEMM with transposed B: C[M,N] = A[M,K] @ B^T[K,N], B stored as [N,K]
void sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K);

// RMSNorm: y = x / rms(x) * gamma
void rmsnorm(const float* x, const float* gamma, float* y,
             int rows, int hidden, float eps);

// Embedding: out[seq_len, hidden] = weight[ids[i]]
void embedding(const float* weight, const int* ids, float* out,
               int seq_len, int hidden);

// RoPE (on-the-fly, no cache table)
// pos_offset: absolute position of the first token (0 for prefill, cached_len for decode)
void apply_rope(const float* x, float* y, int seq_len, int head_dim, float base, int pos_offset = 0);

// Scaled dot-product attention: Q [heads, q_len, head_dim], K/V [heads, kv_len, head_dim]
void sdpa(const float* Q, const float* K, const float* V, float* output,
          int heads, int q_len, int kv_len, int head_dim, bool causal);

// FlashAttention prefill: tiled + online softmax + fused attn@V
// Same API as sdpa but uses O(q_len * head_dim) temp space instead of O(q_len * kv_len)
void flash_sdpa(const float* Q, const float* K, const float* V, float* output,
                int heads, int q_len, int kv_len, int head_dim, bool causal);

// Decode-path SDPA for Q=1: Q [heads, 1, head_dim], K/V [kv_len, kv_hidden]
// Handles GQA: K/V have num_kv_heads columns, expanded to num_heads
void sdpa_decode(const float* Q, const float* K, const float* V, float* output,
                 int num_heads, int num_kv_heads, int head_dim, int kv_len);

// FlashAttention decode: tiled + online softmax for Q=1 with GQA
void flash_sdpa_decode(const float* Q, const float* K, const float* V, float* output,
                       int num_heads, int num_kv_heads, int head_dim, int kv_len);

// Softmax: y[rows, cols]
void softmax(const float* x, float* y, int rows, int cols);

// Generic contiguous transpose over two axes.
void transpose(const float* x, float* y, const int64_t* dims, int rank, int axis0, int axis1);

// Elementwise binary: y[i] = a[i] op b[i], n elements
void add(const float* a, const float* b, float* y, int n);
void mul(const float* a, const float* b, float* y, int n);

// Elementwise unary
void silu(const float* x, float* y, int n);

// Fused: y = silu(gate) * up
void fused_silu_mul(const float* gate, const float* up, float* y, int n);

} // namespace minillm::cpu
