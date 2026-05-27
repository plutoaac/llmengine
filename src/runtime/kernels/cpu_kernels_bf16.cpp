#include "minillm/runtime/kernels/cpu_kernels_bf16.h"

#include "minillm/runtime/kernels/cpu_simd.h"

#include <algorithm>
#include <cstring>

using namespace minillm::simd;

namespace minillm::cpu_bf16 {

// ===========================================================================
// GEMM with transposed B: C[M,N] = A[M,K] @ B^T[N,K], B stored as [N,K]
// A (input) is FP32, B (weight) is BF16, C (output) is FP32.
// Identical structure to cpu::sgemm_nt, but loads B via VF_LOAD_BF16.
// ===========================================================================

void sgemm_nt(const float* A, const bfloat16_t* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        float* c_row = C + m * N;

        int n = 0;
        for (; n + 3 < N; n += 4) {
            const bfloat16_t* b0 = B + (n + 0) * K;
            const bfloat16_t* b1 = B + (n + 1) * K;
            const bfloat16_t* b2 = B + (n + 2) * K;
            const bfloat16_t* b3 = B + (n + 3) * K;

            vfloat vdot0 = VF_SETZERO();
            vfloat vdot1 = VF_SETZERO();
            vfloat vdot2 = VF_SETZERO();
            vfloat vdot3 = VF_SETZERO();

            int k = 0;
            for (; k + MINILLM_SIMD_WIDTH <= K; k += MINILLM_SIMD_WIDTH) {
                vfloat va = VF_LOAD(a_row + k);
                vdot0 = VF_FMADD(va, VF_LOAD_BF16(b0 + k), vdot0);
                vdot1 = VF_FMADD(va, VF_LOAD_BF16(b1 + k), vdot1);
                vdot2 = VF_FMADD(va, VF_LOAD_BF16(b2 + k), vdot2);
                vdot3 = VF_FMADD(va, VF_LOAD_BF16(b3 + k), vdot3);
            }

            float dot0 = hsum(vdot0);
            float dot1 = hsum(vdot1);
            float dot2 = hsum(vdot2);
            float dot3 = hsum(vdot3);
            for (; k < K; ++k) {
                float a = a_row[k];
                dot0 += a * static_cast<float>(b0[k]);
                dot1 += a * static_cast<float>(b1[k]);
                dot2 += a * static_cast<float>(b2[k]);
                dot3 += a * static_cast<float>(b3[k]);
            }

            c_row[n + 0] = dot0;
            c_row[n + 1] = dot1;
            c_row[n + 2] = dot2;
            c_row[n + 3] = dot3;
        }

        for (; n < N; ++n) {
            const bfloat16_t* b_row = B + n * K;
            vfloat vdot = VF_SETZERO();
            int k = 0;
            for (; k + MINILLM_SIMD_WIDTH <= K; k += MINILLM_SIMD_WIDTH) {
                vdot = VF_FMADD(VF_LOAD(a_row + k), VF_LOAD_BF16(b_row + k), vdot);
            }

            float dot = hsum(vdot);
            for (; k < K; ++k) {
                dot += a_row[k] * static_cast<float>(b_row[k]);
            }
            c_row[n] = dot;
        }
    }
}

// ===========================================================================
// GEMM: C[M,N] = A[M,K] @ B[K,N]
// A (input) is FP32, B (weight) is BF16, C (output) is FP32.
// ===========================================================================

void sgemm(const float* A, const bfloat16_t* B, float* C, int M, int N, int K) {
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    constexpr int TILE = 64;
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m += TILE) {
        for (int n = 0; n < N; n += TILE) {
            for (int k = 0; k < K; k += TILE) {
                int tm = std::min(TILE, M - m);
                int tn = std::min(TILE, N - n);
                int tk = std::min(TILE, K - k);

                for (int ii = 0; ii < tm; ++ii) {
                    const float* a_row = A + (m + ii) * K + k;
                    float* c_row = C + (m + ii) * N + n;

                    for (int kk = 0; kk < tk; ++kk) {
                        float a_val = a_row[kk];
                        const bfloat16_t* b_row = B + (k + kk) * N + n;
                        vfloat va = VF_SET1(a_val);

                        int jj = 0;
                        for (; jj + MINILLM_SIMD_WIDTH <= tn; jj += MINILLM_SIMD_WIDTH) {
                            vfloat vc = VF_LOAD(c_row + jj);
                            vfloat vb = VF_LOAD_BF16(b_row + jj);
                            VF_STORE(c_row + jj, VF_FMADD(va, vb, vc));
                        }
                        for (; jj < tn; ++jj) {
                            c_row[jj] += a_val * static_cast<float>(b_row[jj]);
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Embedding: gather rows by id
// weight is BF16, output is FP32.
// ===========================================================================

void embedding(const bfloat16_t* weight, const int* ids, float* out,
               int seq_len, int hidden) {
    for (int s = 0; s < seq_len; ++s) {
        const bfloat16_t* row = weight + ids[s] * hidden;
        float* out_row = out + s * hidden;

        int h = 0;
        for (; h + MINILLM_SIMD_WIDTH <= hidden; h += MINILLM_SIMD_WIDTH)
            VF_STORE(out_row + h, VF_LOAD_BF16(row + h));
        for (; h < hidden; ++h)
            out_row[h] = static_cast<float>(row[h]);
    }
}

} // namespace minillm::cpu_bf16
