#pragma once

#include <cstddef>
#include <cstdint>

#include "minillm/core/status.h"

namespace minillm::cuda {

Status sgemm(const float* A, const float* B, float* C, int M, int N, int K);
Status sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K);
Status add(const float* a, const float* b, float* y, int n);
Status add_bias(float* y, const float* bias, int rows, int cols);
Status mul(const float* a, const float* b, float* y, int n);
Status silu(const float* x, float* y, int n);
Status fused_silu_mul(const float* gate, const float* up, float* y, int n);
Status rmsnorm(const float* x, const float* gamma, float* y,
               int rows, int hidden, float eps);
Status embedding(const float* weight, const int* ids, float* out,
                 int seq_len, int vocab_size, int hidden);
Status apply_rope(const float* x, float* y, int tokens, int num_heads,
                  int head_dim, float base, int pos_offset);
Status softmax(const float* x, float* y, int rows, int cols);
Status transpose(const float* x, float* y, const int64_t* dims,
                 int rank, int axis0, int axis1);
Status sdpa(const float* q, const float* k, const float* v, float* out,
            int batch, int q_len, int num_heads, int num_kv_heads,
            int head_dim, bool causal);

} // namespace minillm::cuda
