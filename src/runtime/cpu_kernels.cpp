#include "minillm/runtime/cpu_kernels.h"

#include "minillm/runtime/cpu_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace minillm::simd;

namespace minillm::cpu {

// ===========================================================================
// GEMM: C[M,N] = A[M,K] @ B[K,N]
// Tiled + SIMD: tile size 64, inner loop vectorized with VF_FMADD
// ===========================================================================

void sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    constexpr int TILE = 64;
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
                        const float* b_row = B + (k + kk) * N + n;
                        vfloat va = VF_SET1(a_val);

                        int jj = 0;
                        for (; jj + MINILLM_SIMD_WIDTH <= tn; jj += MINILLM_SIMD_WIDTH) {
                            vfloat vc = VF_LOAD(c_row + jj);
                            vfloat vb = VF_LOAD(b_row + jj);
                            VF_STORE(c_row + jj, VF_FMADD(va, vb, vc));
                        }
                        for (; jj < tn; ++jj) {
                            c_row[jj] += a_val * b_row[jj];
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// GEMM with transposed B: C[M,N] = A[M,K] @ B^T[K,N], B stored as [N,K]
// Tiled + SIMD: tile size 64, dot-product along K with SIMD
// ===========================================================================

void sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    constexpr int TILE = 64;
    for (int m = 0; m < M; m += TILE) {
        for (int n = 0; n < N; n += TILE) {
            for (int k = 0; k < K; k += TILE) {
                int tm = std::min(TILE, M - m);
                int tn = std::min(TILE, N - n);
                int tk = std::min(TILE, K - k);

                for (int ii = 0; ii < tm; ++ii) {
                    const float* a_row = A + (m + ii) * K + k;
                    float* c_row = C + (m + ii) * N + n;

                    for (int nn = 0; nn < tn; ++nn) {
                        const float* b_row = B + (n + nn) * K + k;

                        vfloat vdot = VF_SETZERO();
                        float dot = 0.0f;
                        int kk = 0;
                        for (; kk + MINILLM_SIMD_WIDTH <= tk; kk += MINILLM_SIMD_WIDTH)
                            vdot = VF_FMADD(VF_LOAD(a_row + kk), VF_LOAD(b_row + kk), vdot);
                        dot = hsum(vdot);
                        for (; kk < tk; ++kk)
                            dot += a_row[kk] * b_row[kk];
                        c_row[nn] += dot;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// RMSNorm: y = x * inv_rms * gamma
// SIMD: vectorized sum-of-squares and elementwise multiply
// ===========================================================================

void rmsnorm(const float* x, const float* gamma, float* y,
             int rows, int hidden, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + r * hidden;
        float* y_row = y + r * hidden;

        // SIMD sum of squares
        vfloat vsq = VF_SETZERO();
        float sum_sq = 0.0f;
        int h = 0;
        for (; h + MINILLM_SIMD_WIDTH <= hidden; h += MINILLM_SIMD_WIDTH) {
            vfloat vx = VF_LOAD(x_row + h);
            vsq = VF_ADD(vsq, VF_MUL(vx, vx));
        }
        sum_sq = hsum(vsq);
        for (; h < hidden; ++h) sum_sq += x_row[h] * x_row[h];

        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden + eps);

        // y = x * inv_rms * gamma
        vfloat vinv = VF_SET1(inv_rms);
        h = 0;
        for (; h + MINILLM_SIMD_WIDTH <= hidden; h += MINILLM_SIMD_WIDTH) {
            vfloat vx = VF_LOAD(x_row + h);
            vfloat vg = VF_LOAD(gamma + h);
            VF_STORE(y_row + h, VF_MUL(VF_MUL(vx, vinv), vg));
        }
        for (; h < hidden; ++h) {
            y_row[h] = x_row[h] * inv_rms * gamma[h];
        }
    }
}

// ===========================================================================
// Embedding: gather rows by id
// ===========================================================================

void embedding(const float* weight, const int* ids, float* out,
               int seq_len, int hidden) {
    for (int s = 0; s < seq_len; ++s) {
        const float* row = weight + ids[s] * hidden;
        float* out_row = out + s * hidden;

        int h = 0;
        for (; h + MINILLM_SIMD_WIDTH <= hidden; h += MINILLM_SIMD_WIDTH)
            VF_STORE(out_row + h, VF_LOAD(row + h));
        for (; h < hidden; ++h) out_row[h] = row[h];
    }
}

// ===========================================================================
// RoPE: rotary position encoding
// SIMD: vectorize the even/odd rotation pairs
// ===========================================================================

void apply_rope(const float* x, float* y, int seq_len, int head_dim, float base) {
    int half = head_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        const float* x_row = x + s * head_dim;
        float* y_row = y + s * head_dim;

        // Precompute cos/sin for this position
        // theta_d = s / base^(2d/head_dim)  — note: this is the on-the-fly version
        // We compute in chunks for SIMD
        int d = 0;
        for (; d + MINILLM_SIMD_WIDTH <= half; d += MINILLM_SIMD_WIDTH) {
            alignas(64) float cos_t[MINILLM_SIMD_WIDTH];
            alignas(64) float sin_t[MINILLM_SIMD_WIDTH];
            for (int k = 0; k < MINILLM_SIMD_WIDTH; ++k) {
                float theta = std::pow(base, -2.0f * (d + k) / head_dim) * s;
                cos_t[k] = std::cos(theta);
                sin_t[k] = std::sin(theta);
            }
            vfloat ve = VF_LOAD(x_row + d);
            vfloat vo = VF_LOAD(x_row + d + half);
            vfloat vc = VF_LOAD(cos_t);
            vfloat vs = VF_LOAD(sin_t);
            VF_STORE(y_row + d, VF_SUB(VF_MUL(ve, vc), VF_MUL(vo, vs)));
            VF_STORE(y_row + d + half, VF_ADD(VF_MUL(ve, vs), VF_MUL(vo, vc)));
        }
        for (; d < half; ++d) {
            float theta = std::pow(base, -2.0f * d / head_dim) * s;
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            float x_even = x_row[d];
            float x_odd = x_row[d + half];
            y_row[d] = x_even * cos_t - x_odd * sin_t;
            y_row[d + half] = x_even * sin_t + x_odd * cos_t;
        }
    }
}

// ===========================================================================
// SDPA: scaled dot-product attention
// SIMD: vectorized QK dot product, softmax, and attn@V
// ===========================================================================

void sdpa(const float* Q, const float* K, const float* V, float* output,
          int heads, int seq_len, int head_dim, bool causal) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int per_head = seq_len * head_dim;

    std::vector<float> scores(static_cast<size_t>(seq_len) * seq_len);

    for (int h = 0; h < heads; ++h) {
        const float* q_h = Q + h * per_head;
        const float* k_h = K + h * per_head;
        const float* v_h = V + h * per_head;
        float* o_h = output + h * per_head;

        // QK^T * scale (SIMD dot product)
        for (int qi = 0; qi < seq_len; ++qi) {
            for (int ki = 0; ki < seq_len; ++ki) {
                if (causal && ki > qi) {
                    scores[qi * seq_len + ki] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                vfloat vsum = VF_SETZERO();
                int d = 0;
                for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                    vsum = VF_ADD(vsum, VF_MUL(
                        VF_LOAD(q_h + qi * head_dim + d),
                        VF_LOAD(k_h + ki * head_dim + d)));
                dot = hsum(vsum);
                for (; d < head_dim; ++d)
                    dot += q_h[qi * head_dim + d] * k_h[ki * head_dim + d];
                scores[qi * seq_len + ki] = dot * scale;
            }
        }

        // Softmax each row (SIMD)
        for (int qi = 0; qi < seq_len; ++qi) {
            float* row = scores.data() + qi * seq_len;

            // Find max
            vfloat vmax = VF_SET1(-std::numeric_limits<float>::max());
            float max_val = -std::numeric_limits<float>::max();
            int ki = 0;
            for (; ki + MINILLM_SIMD_WIDTH <= seq_len; ki += MINILLM_SIMD_WIDTH)
                vmax = VF_MAX(vmax, VF_LOAD(row + ki));
            max_val = hmax(vmax);
            for (; ki < seq_len; ++ki) max_val = std::max(max_val, row[ki]);

            // exp(x - max) and sum
            vfloat vsum = VF_SETZERO();
            float sum = 0.0f;
            ki = 0;
            for (; ki + MINILLM_SIMD_WIDTH <= seq_len; ki += MINILLM_SIMD_WIDTH) {
                vfloat ve = v_exp(VF_SUB(VF_LOAD(row + ki), VF_SET1(max_val)));
                VF_STORE(row + ki, ve);
                vsum = VF_ADD(vsum, ve);
            }
            sum = hsum(vsum);
            for (; ki < seq_len; ++ki) {
                row[ki] = std::exp(row[ki] - max_val);
                sum += row[ki];
            }

            // Normalize
            vfloat vinv = VF_SET1(1.0f / sum);
            ki = 0;
            for (; ki + MINILLM_SIMD_WIDTH <= seq_len; ki += MINILLM_SIMD_WIDTH)
                VF_STORE(row + ki, VF_MUL(VF_LOAD(row + ki), vinv));
            for (; ki < seq_len; ++ki) row[ki] /= sum;
        }

        // attn @ V
        for (int qi = 0; qi < seq_len; ++qi) {
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int ki = 0; ki < seq_len; ++ki)
                    acc += scores[qi * seq_len + ki] * v_h[ki * head_dim + d];
                o_h[qi * head_dim + d] = acc;
            }
        }
    }
}

// ===========================================================================
// Softmax (row-wise, SIMD)
// ===========================================================================

void softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + r * cols;
        float* y_row = y + r * cols;

        // Find max
        vfloat vmax = VF_SET1(-std::numeric_limits<float>::max());
        float max_val = -std::numeric_limits<float>::max();
        int c = 0;
        for (; c + MINILLM_SIMD_WIDTH <= cols; c += MINILLM_SIMD_WIDTH)
            vmax = VF_MAX(vmax, VF_LOAD(x_row + c));
        max_val = hmax(vmax);
        for (; c < cols; ++c) max_val = std::max(max_val, x_row[c]);

        // exp(x - max) and sum
        vfloat vsum = VF_SETZERO();
        float sum = 0.0f;
        c = 0;
        for (; c + MINILLM_SIMD_WIDTH <= cols; c += MINILLM_SIMD_WIDTH) {
            vfloat ve = v_exp(VF_SUB(VF_LOAD(x_row + c), VF_SET1(max_val)));
            VF_STORE(y_row + c, ve);
            vsum = VF_ADD(vsum, ve);
        }
        sum = hsum(vsum);
        for (; c < cols; ++c) {
            y_row[c] = std::exp(x_row[c] - max_val);
            sum += y_row[c];
        }

        // Normalize
        vfloat vinv = VF_SET1(1.0f / sum);
        c = 0;
        for (; c + MINILLM_SIMD_WIDTH <= cols; c += MINILLM_SIMD_WIDTH)
            VF_STORE(y_row + c, VF_MUL(VF_LOAD(y_row + c), vinv));
        for (; c < cols; ++c) y_row[c] /= sum;
    }
}

// ===========================================================================
// Elementwise binary ops (SIMD)
// ===========================================================================

void add(const float* a, const float* b, float* y, int n) {
    int i = 0;
    for (; i + MINILLM_SIMD_WIDTH <= n; i += MINILLM_SIMD_WIDTH)
        VF_STORE(y + i, VF_ADD(VF_LOAD(a + i), VF_LOAD(b + i)));
    for (; i < n; ++i) y[i] = a[i] + b[i];
}

void mul(const float* a, const float* b, float* y, int n) {
    int i = 0;
    for (; i + MINILLM_SIMD_WIDTH <= n; i += MINILLM_SIMD_WIDTH)
        VF_STORE(y + i, VF_MUL(VF_LOAD(a + i), VF_LOAD(b + i)));
    for (; i < n; ++i) y[i] = a[i] * b[i];
}

// ===========================================================================
// SiLU: y = x * sigmoid(x) — SIMD
// ===========================================================================

void silu(const float* x, float* y, int n) {
    int i = 0;
    for (; i + MINILLM_SIMD_WIDTH <= n; i += MINILLM_SIMD_WIDTH) {
        vfloat vx = VF_LOAD(x + i);
        VF_STORE(y + i, VF_MUL(vx, v_sigmoid(vx)));
    }
    for (; i < n; ++i) {
        float val = x[i];
        y[i] = val / (1.0f + std::exp(-val));
    }
}

// ===========================================================================
// Fused SwiGLU: y = silu(gate) * up — SIMD
// ===========================================================================

void fused_silu_mul(const float* gate, const float* up, float* y, int n) {
    int i = 0;
    for (; i + MINILLM_SIMD_WIDTH <= n; i += MINILLM_SIMD_WIDTH) {
        vfloat vg = VF_LOAD(gate + i);
        vfloat vu = VF_LOAD(up + i);
        VF_STORE(y + i, VF_MUL(VF_MUL(vg, v_sigmoid(vg)), vu));
    }
    for (; i < n; ++i) {
        float g = gate[i];
        y[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

} // namespace minillm::cpu
