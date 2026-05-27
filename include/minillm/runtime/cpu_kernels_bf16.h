#pragma once

// BF16-weight kernel declarations.
// These kernels accept BF16 weights and FP32 inputs, producing FP32 outputs.
// This avoids converting BF16 weights to FP32 at load time, halving weight memory.

#include <cstddef>
#include <cstdint>

#include "minillm/utils/bfloat16.hpp"

namespace minillm::cpu_bf16 {

// GEMM with transposed B: C[M,N] = A[M,K] @ B^T[N,K]
// A is FP32 (activation), B is BF16 (weight), C is FP32 (output).
void sgemm_nt(const float* A, const bfloat16_t* B, float* C, int M, int N, int K);

// GEMM: C[M,N] = A[M,K] @ B[K,N]
// A is FP32 (activation), B is BF16 (weight), C is FP32 (output).
void sgemm(const float* A, const bfloat16_t* B, float* C, int M, int N, int K);

// Embedding: out[seq_len, hidden] = weight[ids[i]]
// weight is BF16, output is FP32.
void embedding(const bfloat16_t* weight, const int* ids, float* out,
               int seq_len, int hidden);

} // namespace minillm::cpu_bf16
