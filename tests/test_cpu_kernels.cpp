#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "minillm/runtime/kernels/cpu_kernels.h"

using namespace minillm;

static void assert_near(float actual, float expected, float tol = 1e-5f) {
    if (!(std::abs(actual - expected) <= tol)) {
        std::cerr << "  FAIL: actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << " tol=" << tol << "\n";
        assert(false);
    }
}

static void assert_rel_near(float actual, float expected, float rtol = 1e-4f) {
    float denom = std::max(std::abs(actual), std::abs(expected));
    if (denom < 1e-6f) { assert_near(actual, expected); return; }
    if (!(std::abs(actual - expected) / denom <= rtol)) {
        std::cerr << "  FAIL: actual=" << actual << " expected=" << expected
                  << " rel_diff=" << std::abs(actual - expected) / denom << " rtol=" << rtol << "\n";
        assert(false);
    }
}

static std::vector<float> reference_softmax(const std::vector<float>& x) {
    float max_val = -std::numeric_limits<float>::max();
    for (float v : x) max_val = std::max(max_val, v);

    std::vector<float> y(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - max_val);
        sum += y[i];
    }
    for (float& v : y) v /= sum;
    return y;
}

static std::vector<float> reference_attention_row(
    const float* q, const float* k, const float* v,
    int kv_len, int head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(kv_len));
    for (int i = 0; i < kv_len; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * k[i * head_dim + d];
        }
        scores[static_cast<size_t>(i)] = dot * scale;
    }

    auto probs = reference_softmax(scores);
    std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
    for (int i = 0; i < kv_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            out[static_cast<size_t>(d)] +=
                probs[static_cast<size_t>(i)] * v[i * head_dim + d];
        }
    }
    return out;
}

static std::vector<uint8_t> pack_q8_0_scale1(const std::vector<int8_t>& values) {
    constexpr size_t block_elems = 32;
    constexpr size_t block_bytes = 34;
    const size_t blocks = (values.size() + block_elems - 1) / block_elems;
    std::vector<uint8_t> packed(blocks * block_bytes, 0);

    for (size_t b = 0; b < blocks; ++b) {
        const size_t base = b * block_bytes;
        packed[base + 0] = 0x00;  // fp16 1.0, little-endian
        packed[base + 1] = 0x3c;
        for (size_t i = 0; i < block_elems && b * block_elems + i < values.size(); ++i) {
            packed[base + 2 + i] = static_cast<uint8_t>(values[b * block_elems + i]);
        }
    }
    return packed;
}

void test_sgemm() {
    const float A[] = {1, 2, 3, 4, 5, 6};          // [2, 3]
    const float B[] = {7, 8, 9, 10, 11, 12};       // [3, 2]
    float C[4] = {};

    cpu::sgemm(A, B, C, 2, 2, 3);

    const float expected[] = {58, 64, 139, 154};
    for (int i = 0; i < 4; ++i) assert_near(C[i], expected[i]);
    std::cout << "  PASS test_sgemm\n";
}

void test_sgemm_nt() {
    const float A[] = {1, 2, 3, 4, 5, 6};          // [2, 3]
    const float B[] = {7, 8, 9, 10, 11, 12};       // [2, 3], transposed logical B
    float C[4] = {};

    cpu::sgemm_nt(A, B, C, 2, 2, 3);

    const float expected[] = {50, 68, 122, 167};
    for (int i = 0; i < 4; ++i) assert_near(C[i], expected[i]);
    std::cout << "  PASS test_sgemm_nt\n";
}

void test_sgemm_q8_0() {
    constexpr int M = 2;
    constexpr int K = 2;
    constexpr int N = 32;
    const float A[M * K] = {1.0f, 2.0f, -1.0f, 3.0f};

    std::vector<int8_t> B(static_cast<size_t>(K) * N);
    for (int n = 0; n < N; ++n) {
        B[static_cast<size_t>(n)] = static_cast<int8_t>((n % 9) - 4);
        B[static_cast<size_t>(N + n)] = static_cast<int8_t>((n % 7) - 3);
    }
    auto B_q8 = pack_q8_0_scale1(B);

    float C[M * N] = {};
    cpu::sgemm(A, B_q8.data(), C, M, N, K);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[m * K + k] * static_cast<float>(B[static_cast<size_t>(k) * N + n]);
            }
            assert_near(C[m * N + n], expected);
        }
    }
    std::cout << "  PASS test_sgemm_q8_0\n";
}

void test_sgemm_nt_q8_0() {
    constexpr int M = 2;
    constexpr int N = 3;
    constexpr int K = 32;

    std::vector<float> A(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    }

    std::vector<int8_t> B(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<int8_t>((static_cast<int>(i) * 5) % 17 - 8);
    }
    auto B_q8 = pack_q8_0_scale1(B);

    float C[M * N] = {};
    cpu::sgemm_nt(A.data(), B_q8.data(), C, M, N, K);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[static_cast<size_t>(m) * K + k] *
                    static_cast<float>(B[static_cast<size_t>(n) * K + k]);
            }
            assert_near(C[m * N + n], expected);
        }
    }
    std::cout << "  PASS test_sgemm_nt_q8_0\n";
}

void test_rmsnorm() {
    const float x[] = {1, 2, 3, -1, 0, 1};
    const float gamma[] = {1.0f, 0.5f, 2.0f};
    float y[6] = {};

    cpu::rmsnorm(x, gamma, y, 2, 3, 1e-6f);

    for (int r = 0; r < 2; ++r) {
        float sum_sq = 0.0f;
        for (int h = 0; h < 3; ++h) {
            sum_sq += x[r * 3 + h] * x[r * 3 + h];
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / 3.0f + 1e-6f);
        for (int h = 0; h < 3; ++h) {
            assert_near(y[r * 3 + h], x[r * 3 + h] * inv_rms * gamma[h]);
        }
    }
    std::cout << "  PASS test_rmsnorm\n";
}

void test_silu_swiglu() {
    const float x[] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    const float up[] = {1.0f, 2.0f, -1.0f, 0.5f, 4.0f};
    float silu[5] = {};
    float swiglu[5] = {};

    cpu::silu(x, silu, 5);
    cpu::fused_silu_mul(x, up, swiglu, 5);

    for (int i = 0; i < 5; ++i) {
        const float expected_silu = x[i] / (1.0f + std::exp(-x[i]));
        assert_near(silu[i], expected_silu);
        assert_near(swiglu[i], expected_silu * up[i]);
    }
    std::cout << "  PASS test_silu_swiglu\n";
}

void test_softmax() {
    const float x[] = {1, 2, 3, -1, -1, -1};
    float y[6] = {};

    cpu::softmax(x, y, 2, 3);

    auto row0 = reference_softmax({1, 2, 3});
    for (int i = 0; i < 3; ++i) assert_near(y[i], row0[static_cast<size_t>(i)]);
    for (int i = 3; i < 6; ++i) assert_near(y[i], 1.0f / 3.0f);
    std::cout << "  PASS test_softmax\n";
}

void test_softmax_all_negative_infinity() {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float x[] = {neg_inf, neg_inf, neg_inf, neg_inf};
    float y[4] = {};

    cpu::softmax(x, y, 1, 4);

    for (float v : y) {
        assert(std::isfinite(v));
        assert_near(v, 0.25f);
    }
    std::cout << "  PASS test_softmax_all_negative_infinity\n";
}

void test_embedding_q8_0() {
    constexpr int vocab = 3;
    constexpr int hidden = 32;
    const int ids[] = {2, 0};

    std::vector<int8_t> weight(static_cast<size_t>(vocab) * hidden);
    for (int row = 0; row < vocab; ++row) {
        for (int h = 0; h < hidden; ++h) {
            weight[static_cast<size_t>(row) * hidden + h] =
                static_cast<int8_t>(row * 10 + h - 12);
        }
    }
    auto weight_q8 = pack_q8_0_scale1(weight);

    float out[2 * hidden] = {};
    cpu::embedding(weight_q8.data(), ids, out, 2, hidden);

    for (int s = 0; s < 2; ++s) {
        for (int h = 0; h < hidden; ++h) {
            const int row = ids[s];
            float expected = static_cast<float>(weight[static_cast<size_t>(row) * hidden + h]);
            assert_near(out[s * hidden + h], expected);
        }
    }
    std::cout << "  PASS test_embedding_q8_0\n";
}

void test_rope() {
    const float x[] = {1, 2, 3, 4};
    float y[4] = {};

    cpu::apply_rope(x, y, 1, 4, 10000.0f, 1);

    for (int d = 0; d < 2; ++d) {
        const float theta = std::pow(10000.0f, -2.0f * d / 4.0f);
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        assert_near(y[d], x[d] * c - x[d + 2] * s);
        assert_near(y[d + 2], x[d] * s + x[d + 2] * c);
    }
    std::cout << "  PASS test_rope\n";
}

void test_transpose() {
    const int64_t dims[] = {2, 3, 2};
    const float x[] = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    float y[12] = {};

    cpu::transpose(x, y, dims, 3, 0, 1);

    const float expected[] = {
        1, 2, 7, 8,
        3, 4, 9, 10,
        5, 6, 11, 12,
    };
    for (int i = 0; i < 12; ++i) assert_near(y[i], expected[i]);
    std::cout << "  PASS test_transpose\n";
}

void test_sdpa_causal() {
    const int heads = 1;
    const int q_len = 2;
    const int kv_len = 2;
    const int head_dim = 2;
    const float q[] = {1, 0, 0, 1};
    const float k[] = {1, 0, 0, 1};
    const float v[] = {10, 20, 30, 40};
    float out[4] = {};

    cpu::sdpa(q, k, v, out, heads, q_len, kv_len, head_dim, true);

    assert_near(out[0], 10.0f);
    assert_near(out[1], 20.0f);
    auto expected_row1 = reference_attention_row(q + 2, k, v, 2, 2);
    assert_near(out[2], expected_row1[0]);
    assert_near(out[3], expected_row1[1]);
    std::cout << "  PASS test_sdpa_causal\n";
}

void test_sdpa_decode_gqa() {
    const int num_heads = 2;
    const int num_kv_heads = 1;
    const int head_dim = 2;
    const int kv_len = 3;
    const float q[] = {1, 0, 0, 1};
    const float k[] = {1, 0, 0, 1, 1, 1};
    const float v[] = {10, 0, 0, 20, 30, 40};
    float out[4] = {};

    cpu::sdpa_decode(q, k, v, out, num_heads, num_kv_heads, head_dim, kv_len);

    auto expected_h0 = reference_attention_row(q, k, v, kv_len, head_dim);
    auto expected_h1 = reference_attention_row(q + 2, k, v, kv_len, head_dim);
    assert_near(out[0], expected_h0[0]);
    assert_near(out[1], expected_h0[1]);
    assert_near(out[2], expected_h1[0]);
    assert_near(out[3], expected_h1[1]);
    std::cout << "  PASS test_sdpa_decode_gqa\n";
}

void test_flash_sdpa_matches_naive() {
    // Compare flash_sdpa output against sdpa (naive) for the same inputs
    const int heads = 2, q_len = 4, kv_len = 8, head_dim = 16;

    std::vector<float> Q(static_cast<size_t>(heads) * q_len * head_dim);
    std::vector<float> K(static_cast<size_t>(heads) * kv_len * head_dim);
    std::vector<float> V(static_cast<size_t>(heads) * kv_len * head_dim);
    std::vector<float> out_naive(static_cast<size_t>(heads) * q_len * head_dim);
    std::vector<float> out_flash(static_cast<size_t>(heads) * q_len * head_dim);

    // Fill with small random-ish values (use int arithmetic to avoid size_t underflow)
    for (int i = 0; i < static_cast<int>(Q.size()); ++i) Q[i] = static_cast<float>((i * 7 + 3) % 19 - 9) * 0.1f;
    for (int i = 0; i < static_cast<int>(K.size()); ++i) K[i] = static_cast<float>((i * 11 + 5) % 23 - 11) * 0.1f;
    for (int i = 0; i < static_cast<int>(V.size()); ++i) V[i] = static_cast<float>((i * 13 + 7) % 17 - 8) * 0.1f;

    cpu::sdpa(Q.data(), K.data(), V.data(), out_naive.data(),
              heads, q_len, kv_len, head_dim, true);
    cpu::flash_sdpa(Q.data(), K.data(), V.data(), out_flash.data(),
                    heads, q_len, kv_len, head_dim, true);

    for (size_t i = 0; i < out_naive.size(); ++i) {
        assert_rel_near(out_flash[i], out_naive[i], 1e-3f);
    }

    // Also test non-causal
    std::fill(out_naive.begin(), out_naive.end(), 0.0f);
    std::fill(out_flash.begin(), out_flash.end(), 0.0f);
    cpu::sdpa(Q.data(), K.data(), V.data(), out_naive.data(),
              heads, q_len, kv_len, head_dim, false);
    cpu::flash_sdpa(Q.data(), K.data(), V.data(), out_flash.data(),
                    heads, q_len, kv_len, head_dim, false);

    for (size_t i = 0; i < out_naive.size(); ++i) {
        assert_rel_near(out_flash[i], out_naive[i], 1e-3f);
    }

    std::cout << "  PASS test_flash_sdpa_matches_naive\n";
}

void test_flash_sdpa_larger() {
    // Larger shape to exercise multiple KV tiles
    const int heads = 4, q_len = 16, kv_len = 128, head_dim = 32;

    std::vector<float> Q(static_cast<size_t>(heads) * q_len * head_dim);
    std::vector<float> K(static_cast<size_t>(heads) * kv_len * head_dim);
    std::vector<float> V(static_cast<size_t>(heads) * kv_len * head_dim);
    std::vector<float> out_naive(static_cast<size_t>(heads) * q_len * head_dim);
    std::vector<float> out_flash(static_cast<size_t>(heads) * q_len * head_dim);

    for (int i = 0; i < static_cast<int>(Q.size()); ++i) Q[i] = static_cast<float>((i * 7 + 3) % 19 - 9) * 0.1f;
    for (int i = 0; i < static_cast<int>(K.size()); ++i) K[i] = static_cast<float>((i * 11 + 5) % 23 - 11) * 0.1f;
    for (int i = 0; i < static_cast<int>(V.size()); ++i) V[i] = static_cast<float>((i * 13 + 7) % 17 - 8) * 0.1f;

    cpu::sdpa(Q.data(), K.data(), V.data(), out_naive.data(),
              heads, q_len, kv_len, head_dim, true);
    cpu::flash_sdpa(Q.data(), K.data(), V.data(), out_flash.data(),
                    heads, q_len, kv_len, head_dim, true);

    for (size_t i = 0; i < out_naive.size(); ++i) {
        assert_rel_near(out_flash[i], out_naive[i], 1e-3f);
    }

    std::cout << "  PASS test_flash_sdpa_larger\n";
}

void test_flash_sdpa_decode_matches_naive() {
    // Compare flash_sdpa_decode against sdpa_decode
    const int num_heads = 4, num_kv_heads = 2, head_dim = 16, kv_len = 64;
    int kv_hidden = num_kv_heads * head_dim;

    std::vector<float> Q(static_cast<size_t>(num_heads) * head_dim);
    std::vector<float> K(static_cast<size_t>(kv_len) * kv_hidden);
    std::vector<float> V(static_cast<size_t>(kv_len) * kv_hidden);
    std::vector<float> out_naive(static_cast<size_t>(num_heads) * head_dim);
    std::vector<float> out_flash(static_cast<size_t>(num_heads) * head_dim);

    for (int i = 0; i < static_cast<int>(Q.size()); ++i) Q[i] = static_cast<float>((i * 7 + 3) % 19 - 9) * 0.1f;
    for (int i = 0; i < static_cast<int>(K.size()); ++i) K[i] = static_cast<float>((i * 11 + 5) % 23 - 11) * 0.1f;
    for (int i = 0; i < static_cast<int>(V.size()); ++i) V[i] = static_cast<float>((i * 13 + 7) % 17 - 8) * 0.1f;

    cpu::sdpa_decode(Q.data(), K.data(), V.data(), out_naive.data(),
                     num_heads, num_kv_heads, head_dim, kv_len);
    cpu::flash_sdpa_decode(Q.data(), K.data(), V.data(), out_flash.data(),
                           num_heads, num_kv_heads, head_dim, kv_len);

    for (size_t i = 0; i < out_naive.size(); ++i) {
        assert_rel_near(out_flash[i], out_naive[i], 1e-3f);
    }

    std::cout << "  PASS test_flash_sdpa_decode_matches_naive\n";
}

int main() {
    std::cout << "test_cpu_kernels:\n";
    test_sgemm();
    test_sgemm_nt();
    test_sgemm_q8_0();
    test_sgemm_nt_q8_0();
    test_rmsnorm();
    test_silu_swiglu();
    test_softmax();
    test_softmax_all_negative_infinity();
    test_embedding_q8_0();
    test_rope();
    test_transpose();
    test_sdpa_causal();
    test_sdpa_decode_gqa();
    test_flash_sdpa_matches_naive();
    test_flash_sdpa_larger();
    test_flash_sdpa_decode_matches_naive();
    std::cout << "All tests passed!\n";
    return 0;
}
