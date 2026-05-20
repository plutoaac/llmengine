#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "minillm/runtime/cpu_kernels.h"

using namespace minillm;

static void assert_near(float actual, float expected, float tol = 1e-5f) {
    assert(std::abs(actual - expected) <= tol);
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

int main() {
    std::cout << "test_cpu_kernels:\n";
    test_sgemm();
    test_sgemm_nt();
    test_rmsnorm();
    test_silu_swiglu();
    test_softmax();
    test_rope();
    test_transpose();
    test_sdpa_causal();
    test_sdpa_decode_gqa();
    std::cout << "All tests passed!\n";
    return 0;
}
