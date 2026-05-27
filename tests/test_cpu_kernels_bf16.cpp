#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "minillm/runtime/cpu_kernels.h"
#include "minillm/runtime/cpu_kernels_bf16.h"
#include "minillm/utils/bfloat16.hpp"

using namespace minillm;

static std::mt19937 rng(42);

static void randf(std::vector<float>& v) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& x : v) x = d(rng);
}

static void to_bf16(const std::vector<float>& src, std::vector<bfloat16_t>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = bfloat16_t(src[i]);
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static void assert_near(float actual, float expected, float tol) {
    assert(std::abs(actual - expected) <= tol);
}

// ---- sgemm_nt ----

void test_sgemm_nt_bf16() {
    int M = 4, N = 8, K = 16;
    std::vector<float> hA(M * K), hB(N * K), hC_fp32(M * N), hC_bf16(M * N);
    randf(hA);
    randf(hB);

    std::vector<bfloat16_t> hB_bf16;
    to_bf16(hB, hB_bf16);

    cpu::sgemm_nt(hA.data(), hB.data(), hC_fp32.data(), M, N, K);
    cpu_bf16::sgemm_nt(hA.data(), hB_bf16.data(), hC_bf16.data(), M, N, K);

    float err = max_abs_diff(hC_fp32, hC_bf16);
    // BF16 has ~1e-2 relative error; with K=16, accumulated error ~16 * 1e-2 = ~0.16
    assert(err < 1.0f);
    std::cout << "  PASS test_sgemm_nt_bf16 (max_abs_err=" << err << ")\n";
}

void test_sgemm_nt_bf16_large() {
    int M = 1, N = 4096, K = 4096;
    std::vector<float> hA(M * K), hB(N * K), hC_fp32(M * N), hC_bf16(M * N);
    randf(hA);
    randf(hB);

    std::vector<bfloat16_t> hB_bf16;
    to_bf16(hB, hB_bf16);

    cpu::sgemm_nt(hA.data(), hB.data(), hC_fp32.data(), M, N, K);
    cpu_bf16::sgemm_nt(hA.data(), hB_bf16.data(), hC_bf16.data(), M, N, K);

    // With K=4096, accumulated BF16 error is larger
    float err = max_abs_diff(hC_fp32, hC_bf16);
    // Each dot product has K=4096 terms, each with ~1e-3 relative error
    float max_val = 0.0f;
    for (auto v : hC_fp32) max_val = std::max(max_val, std::abs(v));
    float rel_err = (max_val > 0.0f) ? (err / max_val) : 0.0f;
    assert(rel_err < 0.05f);
    std::cout << "  PASS test_sgemm_nt_bf16_large (max_abs_err=" << err
              << ", rel_err=" << rel_err << ")\n";
}

// ---- sgemm ----

void test_sgemm_bf16() {
    int M = 2, N = 4, K = 8;
    std::vector<float> hA(M * K), hB(K * N), hC_fp32(M * N), hC_bf16(M * N);
    randf(hA);
    randf(hB);

    std::vector<bfloat16_t> hB_bf16;
    to_bf16(hB, hB_bf16);

    cpu::sgemm(hA.data(), hB.data(), hC_fp32.data(), M, N, K);
    cpu_bf16::sgemm(hA.data(), hB_bf16.data(), hC_bf16.data(), M, N, K);

    float err = max_abs_diff(hC_fp32, hC_bf16);
    assert(err < 1.0f);
    std::cout << "  PASS test_sgemm_bf16 (max_abs_err=" << err << ")\n";
}

// ---- embedding ----

void test_embedding_bf16() {
    int vocab = 64, hidden = 32, seq_len = 5;
    std::vector<float> hW_f32(vocab * hidden), hO_fp32(seq_len * hidden), hO_bf16(seq_len * hidden);
    std::vector<int> ids = {0, 5, 10, 63, 42};
    randf(hW_f32);

    std::vector<bfloat16_t> hW_bf16;
    to_bf16(hW_f32, hW_bf16);

    cpu::embedding(hW_f32.data(), ids.data(), hO_fp32.data(), seq_len, hidden);
    cpu_bf16::embedding(hW_bf16.data(), ids.data(), hO_bf16.data(), seq_len, hidden);

    float err = max_abs_diff(hO_fp32, hO_bf16);
    // BF16 round-trip error per element is ~1e-3
    assert(err < 0.01f);
    std::cout << "  PASS test_embedding_bf16 (max_abs_err=" << err << ")\n";
}

// ---- bfloat16_t type ----

void test_bfloat16_t_roundtrip() {
    // float → BF16 → float should round-trip within BF16 precision
    float test_vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, 1000.0f, 0.001f};
    for (float v : test_vals) {
        bfloat16_t bf(v);
        float back = static_cast<float>(bf);
        float err = std::abs(v - back);
        float tol = std::max(std::abs(v) * 0.01f, 1e-6f);
        assert(err < tol);
    }
    std::cout << "  PASS test_bfloat16_t_roundtrip\n";
}

void test_bfloat16_t_special() {
    assert(bfloat16_t(0.0f) == bfloat16_t(0.0f));
    assert(bfloat16_t(1.0f) != bfloat16_t(2.0f));
    assert(bfloat16_t(1.0f) < bfloat16_t(2.0f));
    std::cout << "  PASS test_bfloat16_t_special\n";
}

int main() {
    std::cout << "test_cpu_kernels_bf16:\n";
    test_bfloat16_t_roundtrip();
    test_bfloat16_t_special();
    test_sgemm_nt_bf16();
    test_sgemm_nt_bf16_large();
    test_sgemm_bf16();
    test_embedding_bf16();
    std::cout << "All BF16 kernel tests passed!\n";
    return 0;
}
