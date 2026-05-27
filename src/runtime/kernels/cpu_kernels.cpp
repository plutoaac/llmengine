#include "minillm/runtime/kernels/cpu_kernels.h"

#include "minillm/runtime/kernels/cpu_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace minillm::simd;

namespace minillm::cpu {

// ===========================================================================
// GEMM: C[M,N] = A[M,K] @ B[K,N]
// Tiled + SIMD: tile size 64, inner loop vectorized with VF_FMADD.
// Templated on weight type.
// ===========================================================================

template<typename W>
static void sgemm_impl(const float* A, const W* B, float* C, int M, int N, int K) {
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
                        const W* b_row = B + (k + kk) * N + n;
                        vfloat va = VF_SET1(a_val);

                        int jj = 0;
                        for (; jj + MINILLM_SIMD_WIDTH <= tn; jj += MINILLM_SIMD_WIDTH) {
                            vfloat vc = VF_LOAD(c_row + jj);
                            vfloat vb = load_weight(b_row + jj);
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

void sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    sgemm_impl(A, B, C, M, N, K);
}

void sgemm(const float* A, const bfloat16_t* B, float* C, int M, int N, int K) {
    sgemm_impl(A, B, C, M, N, K);
}

// ===========================================================================
// GEMM with transposed B: C[M,N] = A[M,K] @ B^T[K,N], B stored as [N,K]
// Tiled + SIMD: dot-product along K with SIMD. Templated on weight type.
// ===========================================================================

template<typename W>
static void sgemm_nt_impl(const float* A, const W* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        float* c_row = C + m * N;

        int n = 0;
        for (; n + 3 < N; n += 4) {
            const W* b0 = B + (n + 0) * K;
            const W* b1 = B + (n + 1) * K;
            const W* b2 = B + (n + 2) * K;
            const W* b3 = B + (n + 3) * K;

            vfloat vdot0 = VF_SETZERO();
            vfloat vdot1 = VF_SETZERO();
            vfloat vdot2 = VF_SETZERO();
            vfloat vdot3 = VF_SETZERO();

            int k = 0;
            for (; k + MINILLM_SIMD_WIDTH <= K; k += MINILLM_SIMD_WIDTH) {
                vfloat va = VF_LOAD(a_row + k);
                vdot0 = VF_FMADD(va, load_weight(b0 + k), vdot0);
                vdot1 = VF_FMADD(va, load_weight(b1 + k), vdot1);
                vdot2 = VF_FMADD(va, load_weight(b2 + k), vdot2);
                vdot3 = VF_FMADD(va, load_weight(b3 + k), vdot3);
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
            const W* b_row = B + n * K;
            vfloat vdot = VF_SETZERO();
            int k = 0;
            for (; k + MINILLM_SIMD_WIDTH <= K; k += MINILLM_SIMD_WIDTH) {
                vdot = VF_FMADD(VF_LOAD(a_row + k), load_weight(b_row + k), vdot);
            }

            float dot = hsum(vdot);
            for (; k < K; ++k) {
                dot += a_row[k] * static_cast<float>(b_row[k]);
            }
            c_row[n] = dot;
        }
    }
}

void sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    sgemm_nt_impl(A, B, C, M, N, K);
}

void sgemm_nt(const float* A, const bfloat16_t* B, float* C, int M, int N, int K) {
    sgemm_nt_impl(A, B, C, M, N, K);
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
// Embedding: gather rows by id. Templated on weight type.
// ===========================================================================

template<typename W>
static void embedding_impl(const W* weight, const int* ids, float* out,
                           int seq_len, int hidden) {
    for (int s = 0; s < seq_len; ++s) {
        const W* row = weight + ids[s] * hidden;
        float* out_row = out + s * hidden;

        int h = 0;
        for (; h + MINILLM_SIMD_WIDTH <= hidden; h += MINILLM_SIMD_WIDTH)
            VF_STORE(out_row + h, load_weight(row + h));
        for (; h < hidden; ++h) out_row[h] = static_cast<float>(row[h]);
    }
}

void embedding(const float* weight, const int* ids, float* out,
               int seq_len, int hidden) {
    embedding_impl(weight, ids, out, seq_len, hidden);
}

void embedding(const bfloat16_t* weight, const int* ids, float* out,
               int seq_len, int hidden) {
    embedding_impl(weight, ids, out, seq_len, hidden);
}

// ===========================================================================
// RoPE: rotary position encoding
// SIMD: vectorize the even/odd rotation pairs
// ===========================================================================

void apply_rope(const float* x, float* y, int seq_len, int head_dim, float base, int pos_offset) {
    int half = head_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        const float* x_row = x + s * head_dim;
        float* y_row = y + s * head_dim;

        int pos = pos_offset + s;
        // Precompute cos/sin for this position
        int d = 0;
        for (; d + MINILLM_SIMD_WIDTH <= half; d += MINILLM_SIMD_WIDTH) {
            alignas(64) float cos_t[MINILLM_SIMD_WIDTH];
            alignas(64) float sin_t[MINILLM_SIMD_WIDTH];
            for (int k = 0; k < MINILLM_SIMD_WIDTH; ++k) {
                float theta = std::pow(base, -2.0f * (d + k) / head_dim) * pos;
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
            float theta = std::pow(base, -2.0f * d / head_dim) * pos;
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
// SDPA: scaled dot-product attention (general q_len x kv_len)
// SIMD: vectorized QK dot product, softmax, and attn@V
// ===========================================================================

void sdpa(const float* Q, const float* K, const float* V, float* output,
          int heads, int q_len, int kv_len, int head_dim, bool causal) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int q_per_head = q_len * head_dim;
    int kv_per_head = kv_len * head_dim;

    std::vector<float> scores(static_cast<size_t>(q_len) * kv_len);

    for (int h = 0; h < heads; ++h) {
        const float* q_h = Q + h * q_per_head;
        const float* k_h = K + h * kv_per_head;
        const float* v_h = V + h * kv_per_head;
        float* o_h = output + h * q_per_head;

        // QK^T * scale (SIMD dot product)
        for (int qi = 0; qi < q_len; ++qi) {
            for (int ki = 0; ki < kv_len; ++ki) {
                if (causal && ki > qi) {
                    scores[qi * kv_len + ki] = -std::numeric_limits<float>::infinity();
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
                scores[qi * kv_len + ki] = dot * scale;
            }
        }

        // Softmax each row (SIMD)
        for (int qi = 0; qi < q_len; ++qi) {
            float* row = scores.data() + qi * kv_len;

            // Find max
            vfloat vmax = VF_SET1(-std::numeric_limits<float>::max());
            float max_val = -std::numeric_limits<float>::max();
            int ki = 0;
            for (; ki + MINILLM_SIMD_WIDTH <= kv_len; ki += MINILLM_SIMD_WIDTH)
                vmax = VF_MAX(vmax, VF_LOAD(row + ki));
            max_val = hmax(vmax);
            for (; ki < kv_len; ++ki) max_val = std::max(max_val, row[ki]);

            // exp(x - max) and sum
            vfloat vsum = VF_SETZERO();
            float sum = 0.0f;
            ki = 0;
            for (; ki + MINILLM_SIMD_WIDTH <= kv_len; ki += MINILLM_SIMD_WIDTH) {
                vfloat ve = v_exp(VF_SUB(VF_LOAD(row + ki), VF_SET1(max_val)));
                VF_STORE(row + ki, ve);
                vsum = VF_ADD(vsum, ve);
            }
            sum = hsum(vsum);
            for (; ki < kv_len; ++ki) {
                row[ki] = std::exp(row[ki] - max_val);
                sum += row[ki];
            }

            // Normalize
            if (!std::isfinite(sum) || sum <= 0.0f) {
                // All positions masked: uniform distribution
                float uniform = 1.0f / static_cast<float>(kv_len);
                for (int j = 0; j < kv_len; ++j) row[j] = uniform;
            } else {
                vfloat vinv = VF_SET1(1.0f / sum);
                ki = 0;
                for (; ki + MINILLM_SIMD_WIDTH <= kv_len; ki += MINILLM_SIMD_WIDTH)
                    VF_STORE(row + ki, VF_MUL(VF_LOAD(row + ki), vinv));
                for (; ki < kv_len; ++ki) row[ki] /= sum;
            }
        }

        // attn @ V
        for (int qi = 0; qi < q_len; ++qi) {
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int ki = 0; ki < kv_len; ++ki)
                    acc += scores[qi * kv_len + ki] * v_h[ki * head_dim + d];
                o_h[qi * head_dim + d] = acc;
            }
        }
    }
}

// ===========================================================================
// SDPA decode: Q=1 path with GQA support
// Q: [num_heads, 1, head_dim]   — interleaved heads
// K: [kv_len, kv_hidden]        — cached K, interleaved kv_heads per position
// V: [kv_len, kv_hidden]        — cached V, same layout
// output: [num_heads * head_dim] — single position output
// ===========================================================================

void sdpa_decode(const float* Q, const float* K, const float* V, float* output,
                 int num_heads, int num_kv_heads, int head_dim, int kv_len) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int kv_hidden = num_kv_heads * head_dim;
    int group_size = num_heads / num_kv_heads;

    // For each query head, compute attention over the full KV cache
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_size;
        const float* q_vec = Q + h * head_dim;  // [head_dim]
        const float* k_cache = K;                // [kv_len, kv_hidden]
        const float* v_cache = V;                // [kv_len, kv_hidden]

        // Compute QK^T: [kv_len]
        std::vector<float> scores(kv_len);
        float max_val = -std::numeric_limits<float>::max();

        for (int ki = 0; ki < kv_len; ++ki) {
            const float* k_vec = k_cache + ki * kv_hidden + kv_h * head_dim;
            float dot = 0.0f;
            vfloat vsum = VF_SETZERO();
            int d = 0;
            for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                vsum = VF_ADD(vsum, VF_MUL(
                    VF_LOAD(q_vec + d), VF_LOAD(k_vec + d)));
            dot = hsum(vsum);
            for (; d < head_dim; ++d)
                dot += q_vec[d] * k_vec[d];
            scores[ki] = dot * scale;
            max_val = std::max(max_val, scores[ki]);
        }

        // Softmax
        float sum = 0.0f;
        for (int ki = 0; ki < kv_len; ++ki) {
            scores[ki] = std::exp(scores[ki] - max_val);
            sum += scores[ki];
        }
        // Softmax normalize
        if (!std::isfinite(sum) || sum <= 0.0f) {
            float uniform = 1.0f / static_cast<float>(kv_len);
            for (int ki = 0; ki < kv_len; ++ki) scores[ki] = uniform;
        } else {
            float inv_sum = 1.0f / sum;
            for (int ki = 0; ki < kv_len; ++ki)
                scores[ki] *= inv_sum;
        }

        // Weighted sum of V
        float* o_vec = output + h * head_dim;
        std::memset(o_vec, 0, head_dim * sizeof(float));
        for (int ki = 0; ki < kv_len; ++ki) {
            const float* v_vec = v_cache + ki * kv_hidden + kv_h * head_dim;
            float w = scores[ki];
            vfloat vw = VF_SET1(w);
            int d = 0;
            for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                VF_STORE(o_vec + d, VF_ADD(VF_LOAD(o_vec + d), VF_MUL(vw, VF_LOAD(v_vec + d))));
            for (; d < head_dim; ++d)
                o_vec[d] += w * v_vec[d];
        }
    }
}

// ===========================================================================
// FlashAttention prefill: tiled + online softmax + fused attn@V
// Key insight: instead of materializing the full score matrix (O(q_len * kv_len)),
// we tile over KV and maintain running softmax state (m_i, l_i) per query position.
// This fuses QK^T + softmax + attn@V into one pass with O(q_len * head_dim) temp.
// ===========================================================================

void flash_sdpa(const float* Q, const float* K, const float* V, float* output,
                int heads, int q_len, int kv_len, int head_dim, bool causal) {
    constexpr int KVTILE = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int q_per_head = q_len * head_dim;
    int kv_per_head = kv_len * head_dim;

    for (int h = 0; h < heads; ++h) {
        const float* q_h = Q + h * q_per_head;
        const float* k_h = K + h * kv_per_head;
        const float* v_h = V + h * kv_per_head;
        float* o_h = output + h * q_per_head;

        for (int qi = 0; qi < q_len; ++qi) {
            const float* q_vec = q_h + qi * head_dim;
            float* o_vec = o_h + qi * head_dim;

            // Online softmax state
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            for (int d = 0; d < head_dim; ++d) o_vec[d] = 0.0f;

            // Tile over KV positions
            for (int kt = 0; kt < kv_len; kt += KVTILE) {
                int tile_end = std::min(kt + KVTILE, kv_len);
                // Causal: skip tiles entirely past qi
                if (causal && kt > qi) break;

                // 1. Compute QK^T for this tile, find local max
                float tile_max = -std::numeric_limits<float>::infinity();
                alignas(64) float tile_scores[KVTILE];

                for (int ki = kt; ki < tile_end; ++ki) {
                    if (causal && ki > qi) {
                        tile_scores[ki - kt] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    // SIMD dot product: q_vec . k_row
                    const float* k_row = k_h + ki * head_dim;
                    float dot = 0.0f;
                    vfloat vsum = VF_SETZERO();
                    int d = 0;
                    for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                        vsum = VF_ADD(vsum, VF_MUL(VF_LOAD(q_vec + d), VF_LOAD(k_row + d)));
                    dot = hsum(vsum);
                    for (; d < head_dim; ++d)
                        dot += q_vec[d] * k_row[d];
                    tile_scores[ki - kt] = dot * scale;
                    tile_max = std::max(tile_max, tile_scores[ki - kt]);
                }

                // 2. Online softmax: update running max
                float m_new = std::max(m, tile_max);

                // 3. Correction factor for previous accumulations
                float alpha = (m == -std::numeric_limits<float>::infinity())
                                  ? 1.0f
                                  : std::exp(m - m_new);

                // 4. Exp and compute tile sum
                float tile_sum = 0.0f;
                for (int ki = kt; ki < tile_end; ++ki) {
                    float e = std::exp(tile_scores[ki - kt] - m_new);
                    tile_scores[ki - kt] = e;
                    tile_sum += e;
                }

                // 5. Correct previous output and running sum
                vfloat valpha = VF_SET1(alpha);
                int d = 0;
                for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                    VF_STORE(o_vec + d, VF_MUL(VF_LOAD(o_vec + d), valpha));
                for (; d < head_dim; ++d) o_vec[d] *= alpha;
                l *= alpha;

                // 6. Accumulate weighted V for this tile
                for (int ki = kt; ki < tile_end; ++ki) {
                    float w = tile_scores[ki - kt];
                    if (w == 0.0f) continue;
                    const float* v_row = v_h + ki * head_dim;
                    vfloat vw = VF_SET1(w);
                    d = 0;
                    for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                        VF_STORE(o_vec + d, VF_FMADD(vw, VF_LOAD(v_row + d), VF_LOAD(o_vec + d)));
                    for (; d < head_dim; ++d)
                        o_vec[d] += w * v_row[d];
                }
                l += tile_sum;
                m = m_new;
            }

            // 7. Final normalization
            if (l > 0.0f) {
                float inv_l = 1.0f / l;
                vfloat vinv = VF_SET1(inv_l);
                int d = 0;
                for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                    VF_STORE(o_vec + d, VF_MUL(VF_LOAD(o_vec + d), vinv));
                for (; d < head_dim; ++d) o_vec[d] *= inv_l;
            }
        }
    }
}

// ===========================================================================
// FlashAttention decode: tiled + online softmax for Q=1 with GQA
// Q: [num_heads, 1, head_dim]    — interleaved heads
// K: [kv_len, kv_hidden]         — cached K, interleaved kv_heads per position
// V: [kv_len, kv_hidden]         — cached V, same layout
// output: [num_heads * head_dim]  — single position output
// ===========================================================================

void flash_sdpa_decode(const float* Q, const float* K, const float* V, float* output,
                       int num_heads, int num_kv_heads, int head_dim, int kv_len) {
    constexpr int KVTILE = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int kv_hidden = num_kv_heads * head_dim;
    int group_size = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_size;
        const float* q_vec = Q + h * head_dim;
        float* o_vec = output + h * head_dim;

        // Online softmax state
        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;
        for (int d = 0; d < head_dim; ++d) o_vec[d] = 0.0f;

        // Tile over KV cache
        for (int kt = 0; kt < kv_len; kt += KVTILE) {
            int tile_end = std::min(kt + KVTILE, kv_len);

            // 1. Compute QK^T for this tile, find local max
            float tile_max = -std::numeric_limits<float>::infinity();
            alignas(64) float tile_scores[KVTILE];

            for (int ki = kt; ki < tile_end; ++ki) {
                const float* k_row = K + ki * kv_hidden + kv_h * head_dim;
                float dot = 0.0f;
                vfloat vsum = VF_SETZERO();
                int d = 0;
                for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                    vsum = VF_ADD(vsum, VF_MUL(VF_LOAD(q_vec + d), VF_LOAD(k_row + d)));
                dot = hsum(vsum);
                for (; d < head_dim; ++d)
                    dot += q_vec[d] * k_row[d];
                tile_scores[ki - kt] = dot * scale;
                tile_max = std::max(tile_max, tile_scores[ki - kt]);
            }

            // 2. Online softmax update
            float m_new = std::max(m, tile_max);
            float alpha = (m == -std::numeric_limits<float>::infinity())
                              ? 1.0f
                              : std::exp(m - m_new);

            // 3. Exp and tile sum
            float tile_sum = 0.0f;
            for (int ki = kt; ki < tile_end; ++ki) {
                float e = std::exp(tile_scores[ki - kt] - m_new);
                tile_scores[ki - kt] = e;
                tile_sum += e;
            }

            // 4. Correct previous output
            vfloat valpha = VF_SET1(alpha);
            int d = 0;
            for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                VF_STORE(o_vec + d, VF_MUL(VF_LOAD(o_vec + d), valpha));
            for (; d < head_dim; ++d) o_vec[d] *= alpha;
            l *= alpha;

            // 5. Accumulate weighted V
            for (int ki = kt; ki < tile_end; ++ki) {
                float w = tile_scores[ki - kt];
                if (w == 0.0f) continue;
                const float* v_row = V + ki * kv_hidden + kv_h * head_dim;
                vfloat vw = VF_SET1(w);
                d = 0;
                for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                    VF_STORE(o_vec + d, VF_FMADD(vw, VF_LOAD(v_row + d), VF_LOAD(o_vec + d)));
                for (; d < head_dim; ++d)
                    o_vec[d] += w * v_row[d];
            }
            l += tile_sum;
            m = m_new;
        }

        // 6. Final normalization
        if (l > 0.0f) {
            float inv_l = 1.0f / l;
            vfloat vinv = VF_SET1(inv_l);
            int d = 0;
            for (; d + MINILLM_SIMD_WIDTH <= head_dim; d += MINILLM_SIMD_WIDTH)
                VF_STORE(o_vec + d, VF_MUL(VF_LOAD(o_vec + d), vinv));
            for (; d < head_dim; ++d) o_vec[d] *= inv_l;
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
        if (!std::isfinite(sum) || sum <= 0.0f) {
            float uniform = 1.0f / static_cast<float>(cols);
            for (int j = 0; j < cols; ++j) y_row[j] = uniform;
        } else {
            vfloat vinv = VF_SET1(1.0f / sum);
            c = 0;
            for (; c + MINILLM_SIMD_WIDTH <= cols; c += MINILLM_SIMD_WIDTH)
                VF_STORE(y_row + c, VF_MUL(VF_LOAD(y_row + c), vinv));
            for (; c < cols; ++c) y_row[c] /= sum;
        }
    }
}

// ===========================================================================
// Generic contiguous transpose
// ===========================================================================

void transpose(const float* x, float* y, const int64_t* dims, int rank, int axis0, int axis1) {
    if (axis0 == axis1) {
        size_t total = 1;
        for (int i = 0; i < rank; ++i) total *= static_cast<size_t>(dims[i]);
        std::memcpy(y, x, total * sizeof(float));
        return;
    }

    std::vector<size_t> in_strides(static_cast<size_t>(rank), 1);
    std::vector<size_t> out_dims(static_cast<size_t>(rank));
    std::vector<size_t> out_strides(static_cast<size_t>(rank), 1);

    for (int i = 0; i < rank; ++i) {
        out_dims[static_cast<size_t>(i)] = static_cast<size_t>(dims[i]);
    }
    std::swap(out_dims[static_cast<size_t>(axis0)], out_dims[static_cast<size_t>(axis1)]);

    for (int i = rank - 2; i >= 0; --i) {
        in_strides[static_cast<size_t>(i)] =
            in_strides[static_cast<size_t>(i + 1)] * static_cast<size_t>(dims[i + 1]);
        out_strides[static_cast<size_t>(i)] =
            out_strides[static_cast<size_t>(i + 1)] * out_dims[static_cast<size_t>(i + 1)];
    }

    size_t total = out_dims.empty() ? 0 : out_dims[0] * out_strides[0];
    std::vector<size_t> out_index(static_cast<size_t>(rank));
    std::vector<size_t> in_index(static_cast<size_t>(rank));

    for (size_t linear = 0; linear < total; ++linear) {
        size_t rem = linear;
        for (int i = 0; i < rank; ++i) {
            const size_t stride = out_strides[static_cast<size_t>(i)];
            out_index[static_cast<size_t>(i)] = rem / stride;
            rem %= stride;
        }

        in_index = out_index;
        std::swap(in_index[static_cast<size_t>(axis0)], in_index[static_cast<size_t>(axis1)]);

        size_t src = 0;
        for (int i = 0; i < rank; ++i) {
            src += in_index[static_cast<size_t>(i)] * in_strides[static_cast<size_t>(i)];
        }
        y[linear] = x[src];
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
