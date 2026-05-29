#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "minillm/minillm.h"

using namespace minillm;

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        assert(false);
    }
}

static void check_status(const Status& st) {
    if (!st.ok()) {
        std::cerr << st.to_string() << "\n";
        assert(false);
    }
}

static void sync_ok() {
    check_cuda(cudaDeviceSynchronize());
}

static void assert_near(float actual, float expected, float tol = 1e-4f) {
    assert(std::abs(actual - expected) <= tol);
}

static uint16_t bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static std::vector<uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = bf16_bits(values[i]);
    return out;
}

static std::vector<uint8_t> pack_q8_0(const std::vector<int8_t>& values,
                                      uint16_t scale_bits) {
    constexpr size_t block_elems = 32;
    constexpr size_t block_bytes = 34;
    const size_t blocks = (values.size() + block_elems - 1) / block_elems;
    std::vector<uint8_t> out(blocks * block_bytes, 0);
    for (size_t b = 0; b < blocks; ++b) {
        const size_t base = b * block_bytes;
        out[base + 0] = static_cast<uint8_t>(scale_bits & 0xffu);
        out[base + 1] = static_cast<uint8_t>(scale_bits >> 8);
        for (size_t i = 0; i < block_elems && b * block_elems + i < values.size(); ++i) {
            out[base + 2 + i] = static_cast<uint8_t>(values[b * block_elems + i]);
        }
    }
    return out;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : n_(n) {
        check_cuda(cudaMalloc(&ptr_, n_ * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    void copy_from(const std::vector<T>& host) {
        assert(host.size() == n_);
        check_cuda(cudaMemcpy(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> host(n_);
        check_cuda(cudaMemcpy(host.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T* ptr_{nullptr};
    size_t n_{0};
};

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

static std::vector<float> reference_sdpa(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    int batch, int q_len, int num_heads, int num_kv_heads, int head_dim, bool causal) {
    std::vector<float> out(q.size(), 0.0f);
    const int group_size = num_heads / num_kv_heads;
    const int q_hidden = num_heads * head_dim;
    const int kv_hidden = num_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int b = 0; b < batch; ++b) {
        for (int qi = 0; qi < q_len; ++qi) {
            for (int h = 0; h < num_heads; ++h) {
                const int kv_h = h / group_size;
                const float* q_vec = q.data() + b * q_len * q_hidden +
                                     qi * q_hidden + h * head_dim;

                std::vector<float> scores(static_cast<size_t>(q_len),
                                          -std::numeric_limits<float>::infinity());
                for (int ki = 0; ki < q_len; ++ki) {
                    if (causal && ki > qi) continue;
                    const float* k_vec = k.data() + b * q_len * kv_hidden +
                                         ki * kv_hidden + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    scores[static_cast<size_t>(ki)] = dot * scale;
                }

                auto probs = reference_softmax(scores);
                float* out_vec = out.data() + b * q_len * q_hidden +
                                 qi * q_hidden + h * head_dim;
                for (int ki = 0; ki < q_len; ++ki) {
                    if (causal && ki > qi) continue;
                    const float* v_vec = v.data() + b * q_len * kv_hidden +
                                         ki * kv_hidden + kv_h * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        out_vec[d] += probs[static_cast<size_t>(ki)] * v_vec[d];
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<float> reference_paged_decode(
    const std::vector<float>& q, const std::vector<float>& k_cache,
    const std::vector<float>& v_cache, const std::vector<int>& block_table,
    int seq_len, int num_heads, int num_kv_heads, int head_dim, int block_size) {
    std::vector<float> out(static_cast<size_t>(num_heads) * head_dim, 0.0f);
    const int group_size = num_heads / num_kv_heads;
    const int kv_hidden = num_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / group_size;
        const float* q_vec = q.data() + h * head_dim;
        std::vector<float> scores(static_cast<size_t>(seq_len));

        for (int pos = 0; pos < seq_len; ++pos) {
            const int physical_block = block_table[static_cast<size_t>(pos / block_size)];
            const int block_offset = pos % block_size;
            const size_t base =
                (static_cast<size_t>(physical_block) * block_size + block_offset) * kv_hidden +
                static_cast<size_t>(kv_h) * head_dim;
            const float* k_vec = k_cache.data() + base;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            scores[static_cast<size_t>(pos)] = dot * scale;
        }

        auto probs = reference_softmax(scores);
        float* out_vec = out.data() + h * head_dim;
        for (int pos = 0; pos < seq_len; ++pos) {
            const int physical_block = block_table[static_cast<size_t>(pos / block_size)];
            const int block_offset = pos % block_size;
            const size_t base =
                (static_cast<size_t>(physical_block) * block_size + block_offset) * kv_hidden +
                static_cast<size_t>(kv_h) * head_dim;
            const float* v_vec = v_cache.data() + base;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += probs[static_cast<size_t>(pos)] * v_vec[d];
            }
        }
    }
    return out;
}

void test_cuda_elementwise() {
    const std::vector<float> a{1.0f, -2.0f, 3.5f, 0.0f, 5.0f};
    const std::vector<float> b{2.0f, 4.0f, -1.5f, 7.0f, 0.5f};
    DeviceBuffer<float> da(a.size()), db(b.size()), dy(a.size());
    da.copy_from(a);
    db.copy_from(b);

    check_status(cuda::add(da.get(), db.get(), dy.get(), static_cast<int>(a.size())));
    sync_ok();
    auto y = dy.copy_to_host();
    for (size_t i = 0; i < a.size(); ++i) assert_near(y[i], a[i] + b[i]);

    check_status(cuda::mul(da.get(), db.get(), dy.get(), static_cast<int>(a.size())));
    sync_ok();
    y = dy.copy_to_host();
    for (size_t i = 0; i < a.size(); ++i) assert_near(y[i], a[i] * b[i]);

    check_status(cuda::silu(da.get(), dy.get(), static_cast<int>(a.size())));
    sync_ok();
    y = dy.copy_to_host();
    for (size_t i = 0; i < a.size(); ++i) {
        assert_near(y[i], a[i] / (1.0f + std::exp(-a[i])));
    }

    check_status(cuda::fused_silu_mul(da.get(), db.get(), dy.get(), static_cast<int>(a.size())));
    sync_ok();
    y = dy.copy_to_host();
    for (size_t i = 0; i < a.size(); ++i) {
        const float silu = a[i] / (1.0f + std::exp(-a[i]));
        assert_near(y[i], silu * b[i]);
    }

    std::cout << "  PASS test_cuda_elementwise\n";
}

void test_cuda_gemm_and_bias() {
    const std::vector<float> A{1, 2, 3, 4, 5, 6};       // [2, 3]
    const std::vector<float> B{7, 8, 9, 10, 11, 12};    // [3, 2]
    const std::vector<float> Bt{7, 8, 9, 10, 11, 12};   // [2, 3]
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dBt(Bt.size()), dC(4);
    dA.copy_from(A);
    dB.copy_from(B);
    dBt.copy_from(Bt);

    check_status(cuda::sgemm(dA.get(), dB.get(), dC.get(), 2, 2, 3));
    sync_ok();
    auto C = dC.copy_to_host();
    const std::vector<float> expected{58, 64, 139, 154};
    for (size_t i = 0; i < expected.size(); ++i) assert_near(C[i], expected[i]);

    check_status(cuda::sgemm_reference(dA.get(), dB.get(), dC.get(), 2, 2, 3));
    sync_ok();
    C = dC.copy_to_host();
    for (size_t i = 0; i < expected.size(); ++i) assert_near(C[i], expected[i]);

    check_status(cuda::sgemm_nt(dA.get(), dBt.get(), dC.get(), 2, 2, 3));
    sync_ok();
    C = dC.copy_to_host();
    const std::vector<float> expected_nt{50, 68, 122, 167};
    for (size_t i = 0; i < expected_nt.size(); ++i) assert_near(C[i], expected_nt[i]);

    check_status(cuda::sgemm_nt_reference(dA.get(), dBt.get(), dC.get(), 2, 2, 3));
    sync_ok();
    C = dC.copy_to_host();
    for (size_t i = 0; i < expected_nt.size(); ++i) assert_near(C[i], expected_nt[i]);

    const std::vector<float> bias{0.5f, -1.0f};
    DeviceBuffer<float> dbias(bias.size());
    dbias.copy_from(bias);
    check_status(cuda::add_bias(dC.get(), dbias.get(), 2, 2));
    sync_ok();
    C = dC.copy_to_host();
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            assert_near(C[static_cast<size_t>(r * 2 + c)],
                        expected_nt[static_cast<size_t>(r * 2 + c)] + bias[static_cast<size_t>(c)]);
        }
    }

    {
        const int M = 3;
        const int N = 5;
        const int K = 4;
        std::vector<float> hA(static_cast<size_t>(M) * K);
        std::vector<float> hB(static_cast<size_t>(K) * N);
        std::vector<float> hBt(static_cast<size_t>(N) * K);
        for (int i = 0; i < M * K; ++i) {
            hA[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.25f;
        }
        for (int i = 0; i < K * N; ++i) {
            hB[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.125f;
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                hBt[static_cast<size_t>(n) * K + k] = hB[static_cast<size_t>(k) * N + n];
            }
        }

        DeviceBuffer<float> dA2(hA.size()), dB2(hB.size()), dBt2(hBt.size()),
            dC2(static_cast<size_t>(M) * N);
        dA2.copy_from(hA);
        dB2.copy_from(hB);
        dBt2.copy_from(hBt);

        check_status(cuda::sgemm(dA2.get(), dB2.get(), dC2.get(), M, N, K));
        sync_ok();
        auto got = dC2.copy_to_host();
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expected_val = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected_val += hA[static_cast<size_t>(m) * K + k] *
                                    hB[static_cast<size_t>(k) * N + n];
                }
                assert_near(got[static_cast<size_t>(m) * N + n], expected_val, 1e-5f);
            }
        }

        check_status(cuda::sgemm_nt(dA2.get(), dBt2.get(), dC2.get(), M, N, K));
        sync_ok();
        got = dC2.copy_to_host();
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expected_val = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected_val += hA[static_cast<size_t>(m) * K + k] *
                                    hBt[static_cast<size_t>(n) * K + k];
                }
                assert_near(got[static_cast<size_t>(m) * N + n], expected_val, 1e-5f);
            }
        }
    }

    std::cout << "  PASS test_cuda_gemm_and_bias\n";
}

void test_cuda_bf16_weight_kernels() {
    const std::vector<float> A{1, 2, 3, 4, 5, 6};
    const std::vector<uint16_t> Bt = to_bf16_bits({7, 8, 9, 10, 11, 12});
    DeviceBuffer<float> dA(A.size()), dC(4);
    DeviceBuffer<uint16_t> dBt(Bt.size());
    dA.copy_from(A);
    dBt.copy_from(Bt);

    check_status(cuda::sgemm_nt(dA.get(), dBt.get(), dC.get(), 2, 2, 3));
    sync_ok();
    auto C = dC.copy_to_host();
    const std::vector<float> expected_nt{50, 68, 122, 167};
    for (size_t i = 0; i < expected_nt.size(); ++i) assert_near(C[i], expected_nt[i]);

    const std::vector<uint16_t> bias = to_bf16_bits({0.5f, -1.0f});
    DeviceBuffer<uint16_t> dbias(bias.size());
    dbias.copy_from(bias);
    check_status(cuda::add_bias(dC.get(), dbias.get(), 2, 2));
    sync_ok();
    C = dC.copy_to_host();
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            assert_near(C[static_cast<size_t>(r * 2 + c)],
                        expected_nt[static_cast<size_t>(r * 2 + c)] +
                        (c == 0 ? 0.5f : -1.0f));
        }
    }

    const std::vector<float> x{1, 2, 3, -1, 0, 1};
    const std::vector<uint16_t> gamma = to_bf16_bits({1.0f, 0.5f, 2.0f});
    DeviceBuffer<float> dx(x.size()), dy(x.size());
    DeviceBuffer<uint16_t> dg(gamma.size());
    dx.copy_from(x);
    dg.copy_from(gamma);
    check_status(cuda::rmsnorm(dx.get(), dg.get(), dy.get(), 2, 3, 1e-6f));
    sync_ok();
    auto y = dy.copy_to_host();
    for (int r = 0; r < 2; ++r) {
        float sum_sq = 0.0f;
        for (int h = 0; h < 3; ++h) sum_sq += x[static_cast<size_t>(r * 3 + h)] * x[static_cast<size_t>(r * 3 + h)];
        const float inv = 1.0f / std::sqrt(sum_sq / 3.0f + 1e-6f);
        const std::vector<float> gamma_f{1.0f, 0.5f, 2.0f};
        for (int h = 0; h < 3; ++h) {
            assert_near(y[static_cast<size_t>(r * 3 + h)],
                        x[static_cast<size_t>(r * 3 + h)] * inv * gamma_f[static_cast<size_t>(h)]);
        }
    }

    const std::vector<uint16_t> weight = to_bf16_bits({1, 2, 3, 4, 5, 6, 7, 8});
    const std::vector<int> ids{2, 0, 3};
    DeviceBuffer<uint16_t> dw(weight.size());
    DeviceBuffer<int> dids(ids.size());
    DeviceBuffer<float> de(6);
    dw.copy_from(weight);
    dids.copy_from(ids);
    check_status(cuda::embedding(dw.get(), dids.get(), de.get(), 3, 4, 2));
    sync_ok();
    y = de.copy_to_host();
    const std::vector<float> expected_embedding{5, 6, 1, 2, 7, 8};
    for (size_t i = 0; i < expected_embedding.size(); ++i) assert_near(y[i], expected_embedding[i]);

    std::cout << "  PASS test_cuda_bf16_weight_kernels\n";
}

void test_cuda_q8_0_weight_kernels() {
    {
        const int M = 1;
        const int N = 2;
        const int K = 32;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<int8_t> B(static_cast<size_t>(N) * K);
        for (int i = 0; i < M * K; ++i) A[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.25f;
        for (int i = 0; i < N * K; ++i) B[static_cast<size_t>(i)] = static_cast<int8_t>((i * 5) % 17 - 8);
        auto packed = pack_q8_0(B, 0x3c00);

        DeviceBuffer<float> dA(A.size()), dC(static_cast<size_t>(M) * N);
        DeviceBuffer<uint8_t> dB(packed.size());
        dA.copy_from(A);
        dB.copy_from(packed);
        check_status(cuda::sgemm_nt(dA.get(), dB.get(), dC.get(), M, N, K));
        sync_ok();
        auto C = dC.copy_to_host();
        for (int n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[static_cast<size_t>(k)] *
                            static_cast<float>(B[static_cast<size_t>(n) * K + k]);
            }
            assert_near(C[static_cast<size_t>(n)], expected, 1e-5f);
        }
    }

    {
        const int M = 2;
        const int N = 3;
        const int K = 64;
        constexpr float scale = 0.5f;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<int8_t> B(static_cast<size_t>(N) * K);
        for (int i = 0; i < M * K; ++i) A[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.125f;
        for (int i = 0; i < N * K; ++i) B[static_cast<size_t>(i)] = static_cast<int8_t>((i * 7) % 23 - 11);
        auto packed = pack_q8_0(B, 0x3800);

        DeviceBuffer<float> dA(A.size()), dC(static_cast<size_t>(M) * N);
        DeviceBuffer<uint8_t> dB(packed.size());
        dA.copy_from(A);
        dB.copy_from(packed);
        check_status(cuda::sgemm_nt(dA.get(), dB.get(), dC.get(), M, N, K));
        sync_ok();
        auto C = dC.copy_to_host();
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected += A[static_cast<size_t>(m) * K + k] *
                                static_cast<float>(B[static_cast<size_t>(n) * K + k]) *
                                scale;
                }
                assert_near(C[static_cast<size_t>(m) * N + n], expected, 1e-4f);
            }
        }
    }

    {
        const int M = 2;
        const int N = 64;
        const int K = 32;
        constexpr float scale = 0.5f;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<int8_t> B(static_cast<size_t>(K) * N);
        for (int i = 0; i < M * K; ++i) A[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.125f;
        for (int i = 0; i < K * N; ++i) B[static_cast<size_t>(i)] = static_cast<int8_t>((i * 3) % 29 - 14);
        auto packed = pack_q8_0(B, 0x3800);

        DeviceBuffer<float> dA(A.size()), dC(static_cast<size_t>(M) * N);
        DeviceBuffer<uint8_t> dB(packed.size());
        dA.copy_from(A);
        dB.copy_from(packed);
        check_status(cuda::sgemm(dA.get(), dB.get(), dC.get(), M, N, K));
        sync_ok();
        auto C = dC.copy_to_host();
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected += A[static_cast<size_t>(m) * K + k] *
                                static_cast<float>(B[static_cast<size_t>(k) * N + n]) *
                                scale;
                }
                assert_near(C[static_cast<size_t>(m) * N + n], expected, 1e-4f);
            }
        }
    }

    {
        const int vocab = 3;
        const int hidden = 64;
        constexpr float scale = 0.5f;
        std::vector<int8_t> W(static_cast<size_t>(vocab) * hidden);
        for (int i = 0; i < vocab * hidden; ++i) {
            W[static_cast<size_t>(i)] = static_cast<int8_t>((i * 5) % 31 - 15);
        }
        auto packed = pack_q8_0(W, 0x3800);
        const std::vector<int> ids{2, 0, 1};

        DeviceBuffer<uint8_t> dW(packed.size());
        DeviceBuffer<int> dids(ids.size());
        DeviceBuffer<float> dout(static_cast<size_t>(ids.size()) * hidden);
        dW.copy_from(packed);
        dids.copy_from(ids);
        check_status(cuda::embedding(dW.get(), dids.get(), dout.get(),
                                     static_cast<int>(ids.size()), vocab, hidden));
        sync_ok();
        auto out = dout.copy_to_host();
        for (size_t s = 0; s < ids.size(); ++s) {
            for (int h = 0; h < hidden; ++h) {
                const float expected =
                    static_cast<float>(W[static_cast<size_t>(ids[s]) * hidden + h]) * scale;
                assert_near(out[s * static_cast<size_t>(hidden) + h], expected, 1e-5f);
            }
        }
    }

    std::cout << "  PASS test_cuda_q8_0_weight_kernels\n";
}

void test_cuda_norm_embedding_rope_softmax_transpose() {
    {
        const std::vector<float> x{1, 2, 3, -1, 0, 1};
        const std::vector<float> gamma{1.0f, 0.5f, 2.0f};
        DeviceBuffer<float> dx(x.size()), dg(gamma.size()), dy(x.size());
        dx.copy_from(x);
        dg.copy_from(gamma);
        check_status(cuda::rmsnorm(dx.get(), dg.get(), dy.get(), 2, 3, 1e-6f));
        sync_ok();
        auto y = dy.copy_to_host();
        for (int r = 0; r < 2; ++r) {
            float sum_sq = 0.0f;
            for (int h = 0; h < 3; ++h) sum_sq += x[static_cast<size_t>(r * 3 + h)] * x[static_cast<size_t>(r * 3 + h)];
            const float inv = 1.0f / std::sqrt(sum_sq / 3.0f + 1e-6f);
            for (int h = 0; h < 3; ++h) {
                assert_near(y[static_cast<size_t>(r * 3 + h)],
                            x[static_cast<size_t>(r * 3 + h)] * inv * gamma[static_cast<size_t>(h)]);
            }
        }
    }

    {
        const std::vector<float> weight{1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<int> ids{2, 0, 3};
        DeviceBuffer<float> dw(weight.size()), dy(6);
        DeviceBuffer<int> dids(ids.size());
        dw.copy_from(weight);
        dids.copy_from(ids);
        check_status(cuda::embedding(dw.get(), dids.get(), dy.get(), 3, 4, 2));
        sync_ok();
        auto y = dy.copy_to_host();
        const std::vector<float> expected{5, 6, 1, 2, 7, 8};
        for (size_t i = 0; i < expected.size(); ++i) assert_near(y[i], expected[i]);
    }

    {
        const std::vector<float> x{
            1, 2, 3, 4, 5, 6, 7, 8,
            -1, -2, 0.5f, 1.5f, 2.5f, 3.5f, -0.5f, 4.5f,
        };
        DeviceBuffer<float> dx(x.size()), dy(x.size());
        dx.copy_from(x);
        check_status(cuda::apply_rope(dx.get(), dy.get(), 2, 2, 4, 10000.0f, 1));
        sync_ok();
        auto y = dy.copy_to_host();
        for (int t = 0; t < 2; ++t) {
            for (int h = 0; h < 2; ++h) {
                const int base = t * 8 + h * 4;
                for (int d = 0; d < 2; ++d) {
                    const float theta = std::pow(10000.0f, -2.0f * d / 4.0f) * (1 + t);
                    const float c = std::cos(theta);
                    const float s = std::sin(theta);
                    assert_near(y[static_cast<size_t>(base + d)],
                                x[static_cast<size_t>(base + d)] * c -
                                x[static_cast<size_t>(base + d + 2)] * s);
                    assert_near(y[static_cast<size_t>(base + d + 2)],
                                x[static_cast<size_t>(base + d)] * s +
                                x[static_cast<size_t>(base + d + 2)] * c);
                }
            }
        }
    }

    {
        const std::vector<float> x{1, 2, 3, -1, -1, -1};
        DeviceBuffer<float> dx(x.size()), dy(x.size());
        dx.copy_from(x);
        check_status(cuda::softmax(dx.get(), dy.get(), 2, 3));
        sync_ok();
        auto y = dy.copy_to_host();
        auto row0 = reference_softmax({1, 2, 3});
        for (int i = 0; i < 3; ++i) assert_near(y[static_cast<size_t>(i)], row0[static_cast<size_t>(i)]);
        for (int i = 3; i < 6; ++i) assert_near(y[static_cast<size_t>(i)], 1.0f / 3.0f);
    }

    {
        const int64_t dims[] = {2, 3, 2};
        const std::vector<float> x{
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        };
        DeviceBuffer<float> dx(x.size()), dy(x.size());
        dx.copy_from(x);
        check_status(cuda::transpose(dx.get(), dy.get(), dims, 3, 0, 1));
        sync_ok();
        auto y = dy.copy_to_host();
        const std::vector<float> expected{
            1, 2, 7, 8,
            3, 4, 9, 10,
            5, 6, 11, 12,
        };
        for (size_t i = 0; i < expected.size(); ++i) assert_near(y[i], expected[i]);
    }

    std::cout << "  PASS test_cuda_norm_embedding_rope_softmax_transpose\n";
}

void test_cuda_sdpa_gqa() {
    const int batch = 1;
    const int q_len = 3;
    const int num_heads = 2;
    const int num_kv_heads = 1;
    const int head_dim = 2;
    const std::vector<float> q{
        1, 0, 0, 1,
        0.5f, 0.5f, 1, -1,
        -0.5f, 1, 1, 1,
    };
    const std::vector<float> k{
        1, 0,
        0, 1,
        1, 1,
    };
    const std::vector<float> v{
        10, 0,
        0, 20,
        30, 40,
    };
    DeviceBuffer<float> dq(q.size()), dk(k.size()), dv(v.size()), dout(q.size());
    dq.copy_from(q);
    dk.copy_from(k);
    dv.copy_from(v);

    check_status(cuda::sdpa(dq.get(), dk.get(), dv.get(), dout.get(),
                            batch, q_len, num_heads, num_kv_heads, head_dim, true));
    sync_ok();
    auto out = dout.copy_to_host();
    auto expected = reference_sdpa(q, k, v, batch, q_len, num_heads, num_kv_heads, head_dim, true);
    for (size_t i = 0; i < expected.size(); ++i) {
        assert_near(out[i], expected[i], 1e-3f);
    }

    {
        const int batch2 = 2;
        const int q_len2 = 37;
        const int num_heads2 = 4;
        const int num_kv_heads2 = 2;
        const int head_dim2 = 64;
        const int q_hidden = num_heads2 * head_dim2;
        const int kv_hidden = num_kv_heads2 * head_dim2;
        std::vector<float> q2(static_cast<size_t>(batch2) * q_len2 * q_hidden);
        std::vector<float> k2(static_cast<size_t>(batch2) * q_len2 * kv_hidden);
        std::vector<float> v2(k2.size());
        for (size_t i = 0; i < q2.size(); ++i) {
            q2[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.03125f;
        }
        for (size_t i = 0; i < k2.size(); ++i) {
            k2[i] = static_cast<float>((static_cast<int>(i * 3) % 19) - 9) * 0.025f;
            v2[i] = static_cast<float>((static_cast<int>(i * 5) % 23) - 11) * 0.05f;
        }

        DeviceBuffer<float> dq2(q2.size()), dk2(k2.size()), dv2(v2.size()), dout2(q2.size());
        dq2.copy_from(q2);
        dk2.copy_from(k2);
        dv2.copy_from(v2);
        check_status(cuda::sdpa(dq2.get(), dk2.get(), dv2.get(), dout2.get(),
                                batch2, q_len2, num_heads2, num_kv_heads2, head_dim2, true));
        sync_ok();
        auto got2 = dout2.copy_to_host();
        auto expected2 = reference_sdpa(q2, k2, v2, batch2, q_len2, num_heads2,
                                        num_kv_heads2, head_dim2, true);
        for (size_t i = 0; i < expected2.size(); ++i) {
            assert_near(got2[i], expected2[i], 2e-3f);
        }
    }

    std::cout << "  PASS test_cuda_sdpa_gqa\n";
}

void test_cuda_paged_attention_decode_gqa() {
    const int seq_len = 5;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 2;
    const int block_size = 2;
    const int max_blocks = 4;
    const int kv_hidden = num_kv_heads * head_dim;
    const std::vector<int> block_table{2, 0, 3};

    const std::vector<float> q{
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        -0.5f, 1.0f,
    };
    std::vector<float> k_cache(
        static_cast<size_t>(max_blocks) * block_size * kv_hidden, -99.0f);
    std::vector<float> v_cache(k_cache.size(), 99.0f);

    for (int pos = 0; pos < seq_len; ++pos) {
        const int physical_block = block_table[static_cast<size_t>(pos / block_size)];
        const int block_offset = pos % block_size;
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t dst =
                    (static_cast<size_t>(physical_block) * block_size + block_offset) *
                    kv_hidden + static_cast<size_t>(h) * head_dim + d;
                k_cache[dst] = 0.1f * static_cast<float>(pos + 1) +
                               0.2f * static_cast<float>(h + 1) +
                               0.05f * static_cast<float>(d);
                v_cache[dst] = static_cast<float>(10 * (h + 1) + pos * 3 + d);
            }
        }
    }

    DeviceBuffer<float> dq(q.size()), dk(k_cache.size()), dv(v_cache.size()),
        dout(static_cast<size_t>(num_heads) * head_dim);
    DeviceBuffer<int> dtable(block_table.size());
    dq.copy_from(q);
    dk.copy_from(k_cache);
    dv.copy_from(v_cache);
    dtable.copy_from(block_table);

    check_status(cuda::paged_attention_decode(
        dq.get(), dk.get(), dv.get(), dtable.get(), dout.get(), seq_len,
        num_heads, num_kv_heads, head_dim, block_size));
    sync_ok();

    auto out = dout.copy_to_host();
    auto expected = reference_paged_decode(q, k_cache, v_cache, block_table,
                                           seq_len, num_heads, num_kv_heads,
                                           head_dim, block_size);
    for (size_t i = 0; i < expected.size(); ++i) {
        assert_near(out[i], expected[i], 1e-3f);
    }

    std::cout << "  PASS test_cuda_paged_attention_decode_gqa\n";
}

void test_cuda_paged_attention_decode_batch_gqa() {
    const int batch_size = 2;
    const int max_blocks_per_sequence = 3;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 2;
    const int block_size = 2;
    const int max_blocks = 5;
    const int kv_hidden = num_kv_heads * head_dim;
    const std::vector<int> sequence_lengths{5, 3};
    const std::vector<int> block_tables{
        2, 0, 3,
        1, 4, -1,
    };

    const std::vector<float> q{
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        -0.5f, 1.0f,

        0.25f, 1.0f,
        1.0f, -0.25f,
        -1.0f, 0.5f,
        0.75f, 0.25f,
    };
    std::vector<float> k_cache(
        static_cast<size_t>(max_blocks) * block_size * kv_hidden, -99.0f);
    std::vector<float> v_cache(k_cache.size(), 99.0f);

    for (int b = 0; b < batch_size; ++b) {
        const int seq_len = sequence_lengths[static_cast<size_t>(b)];
        const int* table =
            block_tables.data() + static_cast<size_t>(b) * max_blocks_per_sequence;
        for (int pos = 0; pos < seq_len; ++pos) {
            const int physical_block = table[pos / block_size];
            const int block_offset = pos % block_size;
            for (int h = 0; h < num_kv_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t dst =
                        (static_cast<size_t>(physical_block) * block_size + block_offset) *
                        kv_hidden + static_cast<size_t>(h) * head_dim + d;
                    k_cache[dst] = 0.07f * static_cast<float>((b + 1) * (pos + 1)) +
                                   0.13f * static_cast<float>(h + 1) +
                                   0.03f * static_cast<float>(d);
                    v_cache[dst] = static_cast<float>(
                        100 * (b + 1) + 10 * (h + 1) + pos * 3 + d);
                }
            }
        }
    }

    DeviceBuffer<float> dq(q.size()), dk(k_cache.size()), dv(v_cache.size()),
        dout(static_cast<size_t>(batch_size) * num_heads * head_dim);
    DeviceBuffer<int> dtables(block_tables.size()), dlengths(sequence_lengths.size());
    dq.copy_from(q);
    dk.copy_from(k_cache);
    dv.copy_from(v_cache);
    dtables.copy_from(block_tables);
    dlengths.copy_from(sequence_lengths);

    check_status(cuda::paged_attention_decode_batch(
        dq.get(), dk.get(), dv.get(), dtables.get(), dlengths.get(), dout.get(),
        batch_size, max_blocks_per_sequence, num_heads, num_kv_heads,
        head_dim, block_size));
    sync_ok();

    auto out = dout.copy_to_host();
    for (int b = 0; b < batch_size; ++b) {
        const size_t q_offset = static_cast<size_t>(b) * num_heads * head_dim;
        std::vector<float> q_seq(q.begin() + static_cast<std::ptrdiff_t>(q_offset),
                                 q.begin() + static_cast<std::ptrdiff_t>(
                                     q_offset + num_heads * head_dim));
        std::vector<int> table(
            block_tables.begin() + static_cast<std::ptrdiff_t>(
                static_cast<size_t>(b) * max_blocks_per_sequence),
            block_tables.begin() + static_cast<std::ptrdiff_t>(
                static_cast<size_t>(b + 1) * max_blocks_per_sequence));
        auto expected = reference_paged_decode(
            q_seq, k_cache, v_cache, table, sequence_lengths[static_cast<size_t>(b)],
            num_heads, num_kv_heads, head_dim, block_size);
        for (size_t i = 0; i < expected.size(); ++i) {
            assert_near(out[q_offset + i], expected[i], 1e-3f);
        }
    }

    std::cout << "  PASS test_cuda_paged_attention_decode_batch_gqa\n";
}

void test_cuda_kv_cache_attention_decode_gqa() {
    const int kv_len = 4;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 2;
    const int kv_hidden = num_kv_heads * head_dim;

    const std::vector<float> q{
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        -0.5f, 1.0f,
    };
    std::vector<float> k_cache(static_cast<size_t>(kv_len) * kv_hidden);
    std::vector<float> v_cache(k_cache.size());
    for (int pos = 0; pos < kv_len; ++pos) {
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t idx =
                    static_cast<size_t>(pos) * kv_hidden +
                    static_cast<size_t>(h) * head_dim + d;
                k_cache[idx] = 0.11f * static_cast<float>(pos + 1) +
                               0.17f * static_cast<float>(h + 1) +
                               0.03f * static_cast<float>(d);
                v_cache[idx] = static_cast<float>(10 * (h + 1) + pos * 4 + d);
            }
        }
    }

    KVCache cache;
    check_status(cache.init_cuda(1, num_kv_heads, head_dim, kv_len));
    check_cuda(cudaMemcpy(cache.k_data(0), k_cache.data(),
                          k_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(cache.v_data(0), v_cache.data(),
                          v_cache.size() * sizeof(float), cudaMemcpyHostToDevice));

    DeviceBuffer<float> dq(q.size()), dout(static_cast<size_t>(num_heads) * head_dim);
    dq.copy_from(q);
    check_status(cuda::kv_cache_attention_decode(
        dq.get(), cache.k_data(0), cache.v_data(0), dout.get(),
        kv_len, num_heads, num_kv_heads, head_dim));
    sync_ok();

    auto out = dout.copy_to_host();
    auto expected = reference_paged_decode(
        q, k_cache, v_cache, std::vector<int>{0}, kv_len, num_heads,
        num_kv_heads, head_dim, kv_len);
    for (size_t i = 0; i < expected.size(); ++i) {
        assert_near(out[i], expected[i], 1e-3f);
    }

    std::cout << "  PASS test_cuda_kv_cache_attention_decode_gqa\n";
}

void test_cuda_executor_linear_bias() {
    Graph graph;
    GraphBuilder gb(graph);
    auto x = gb.input("x", Shape({2, 3}), DType::Float32, Device::cuda(0));
    auto w = gb.constant("w", Shape({2, 3}), DType::Float32, Device::cuda(0));
    auto b = gb.constant("b", Shape({2}), DType::Float32, Device::cuda(0));
    assert(x && w && b);
    auto y = gb.linear(*x, *w, *b, "linear");
    assert(y);
    auto out = gb.output(*y, "out");
    assert(out);

    Tensor tx("x", Shape({2, 3}), DType::Float32, Device::cuda(0));
    Tensor tw("w", Shape({2, 3}), DType::Float32, Device::cuda(0));
    Tensor tb("b", Shape({2}), DType::Float32, Device::cuda(0));
    check_status(tx.allocate_cuda());
    check_status(tw.allocate_cuda());
    check_status(tb.allocate_cuda());

    const std::vector<float> hx{1, 2, 3, 4, 5, 6};
    const std::vector<float> hw{7, 8, 9, 10, 11, 12};
    const std::vector<float> hb{0.5f, -1.0f};
    check_cuda(cudaMemcpy(tx.data(), hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(tw.data(), hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(tb.data(), hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice));

    RuntimeContext ctx;
    check_status(ctx.bind(*x, &tx));
    check_status(ctx.bind(*w, &tw));
    check_status(ctx.bind(*b, &tb));
    check_status(ctx.allocate_intermediates(graph));

    KernelRegistry registry;
    register_cuda_kernels(registry);
    CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
    check_status(executor.compile(graph));
    check_status(executor.run(ctx));
    sync_ok();

    Tensor* y_tensor = ctx.get(*y);
    assert(y_tensor != nullptr);
    std::vector<float> got(4);
    check_cuda(cudaMemcpy(got.data(), y_tensor->data(), got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const std::vector<float> expected{50.5f, 67.0f, 122.5f, 166.0f};
    for (size_t i = 0; i < expected.size(); ++i) assert_near(got[i], expected[i]);

    std::cout << "  PASS test_cuda_executor_linear_bias\n";
}

void test_cuda_executor_q8_0_linear_and_matmul() {
    {
        Graph graph;
        GraphBuilder gb(graph);
        auto x = gb.input("x", Shape({2, 32}), DType::Float32, Device::cuda(0));
        auto w = gb.constant("w", Shape({3, 32}), DType::Float32, Device::cuda(0));
        assert(x && w);
        auto y = gb.linear(*x, *w, std::nullopt, "linear_q8");
        assert(y);
        auto out = gb.output(*y, "out");
        assert(out);

        Tensor tx("x", Shape({2, 32}), DType::Float32, Device::cuda(0));
        Tensor tw("w", Shape({3, 32}), DType::Q8_0, Device::cuda(0));
        check_status(tx.allocate_cuda());

        std::vector<float> hx(64);
        for (int i = 0; i < 64; ++i) hx[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.125f;
        std::vector<int8_t> hw(3 * 32);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = static_cast<int8_t>((static_cast<int>(i) * 7) % 19 - 9);
        auto packed = pack_q8_0(hw, 0x3c00);
        check_status(tw.allocate_cuda_bytes(packed.size()));
        check_cuda(cudaMemcpy(tx.data(), hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(tw.data(), packed.data(), packed.size(), cudaMemcpyHostToDevice));

        RuntimeContext ctx;
        check_status(ctx.bind(*x, &tx));
        check_status(ctx.bind(*w, &tw));
        check_status(ctx.allocate_intermediates(graph));

        KernelRegistry registry;
        register_cuda_kernels(registry);
        CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
        check_status(executor.compile(graph));
        check_status(executor.run(ctx));
        sync_ok();

        Tensor* y_tensor = ctx.get(*y);
        assert(y_tensor != nullptr);
        std::vector<float> got(6);
        check_cuda(cudaMemcpy(got.data(), y_tensor->data(), got.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 3; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < 32; ++k) {
                    expected += hx[static_cast<size_t>(m) * 32 + k] *
                                static_cast<float>(hw[static_cast<size_t>(n) * 32 + k]);
                }
                assert_near(got[static_cast<size_t>(m) * 3 + n], expected, 1e-4f);
            }
        }
    }

    {
        Graph graph;
        GraphBuilder gb(graph);
        auto a = gb.input("a", Shape({2, 32}), DType::Float32, Device::cuda(0));
        auto b = gb.constant("b", Shape({32, 32}), DType::Float32, Device::cuda(0));
        assert(a && b);
        auto y = gb.matmul(*a, *b, "matmul_q8");
        assert(y);
        auto out = gb.output(*y, "out");
        assert(out);

        Tensor ta("a", Shape({2, 32}), DType::Float32, Device::cuda(0));
        Tensor tb("b", Shape({32, 32}), DType::Q8_0, Device::cuda(0));
        check_status(ta.allocate_cuda());

        std::vector<float> ha(64);
        for (int i = 0; i < 64; ++i) ha[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.125f;
        std::vector<int8_t> hb(32 * 32);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<int8_t>((static_cast<int>(i) * 5) % 17 - 8);
        auto packed = pack_q8_0(hb, 0x3c00);
        check_status(tb.allocate_cuda_bytes(packed.size()));
        check_cuda(cudaMemcpy(ta.data(), ha.data(), ha.size() * sizeof(float), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(tb.data(), packed.data(), packed.size(), cudaMemcpyHostToDevice));

        RuntimeContext ctx;
        check_status(ctx.bind(*a, &ta));
        check_status(ctx.bind(*b, &tb));
        check_status(ctx.allocate_intermediates(graph));

        KernelRegistry registry;
        register_cuda_kernels(registry);
        CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
        check_status(executor.compile(graph));
        check_status(executor.run(ctx));
        sync_ok();

        Tensor* y_tensor = ctx.get(*y);
        assert(y_tensor != nullptr);
        std::vector<float> got(64);
        check_cuda(cudaMemcpy(got.data(), y_tensor->data(), got.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 32; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < 32; ++k) {
                    expected += ha[static_cast<size_t>(m) * 32 + k] *
                                static_cast<float>(hb[static_cast<size_t>(k) * 32 + n]);
                }
                assert_near(got[static_cast<size_t>(m) * 32 + n], expected, 1e-4f);
            }
        }
    }

    std::cout << "  PASS test_cuda_executor_q8_0_linear_and_matmul\n";
}

void test_cuda_executor_q8_0_rejects_invalid_inputs() {
    {
        Graph graph;
        GraphBuilder gb(graph);
        auto x = gb.input("x", Shape({1, 31}), DType::Float32, Device::cuda(0));
        auto w = gb.constant("w", Shape({2, 31}), DType::Float32, Device::cuda(0));
        assert(x && w);
        auto y = gb.linear(*x, *w, std::nullopt, "linear_q8_bad_k");
        assert(y);

        Tensor tx("x", Shape({1, 31}), DType::Float32, Device::cuda(0));
        Tensor tw("w", Shape({2, 31}), DType::Q8_0, Device::cuda(0));
        check_status(tx.allocate_cuda());
        std::vector<int8_t> hw(2 * 31, 1);
        auto packed = pack_q8_0(hw, 0x3c00);
        check_status(tw.allocate_cuda_bytes(packed.size()));

        RuntimeContext ctx;
        check_status(ctx.bind(*x, &tx));
        check_status(ctx.bind(*w, &tw));
        check_status(ctx.allocate_intermediates(graph));

        KernelRegistry registry;
        register_cuda_kernels(registry);
        CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
        check_status(executor.compile(graph));
        auto st = executor.run(ctx);
        assert(!st.ok());
        assert(std::string(st.message()).find("Q8_0") != std::string::npos);
    }

    {
        Graph graph;
        GraphBuilder gb(graph);
        auto x = gb.input("x", Shape({1, 32}), DType::Float32, Device::cuda(0));
        auto w = gb.constant("w", Shape({1, 32}), DType::Float32, Device::cuda(0));
        assert(x && w);
        auto y = gb.linear(*x, *w, std::nullopt, "linear_q8_short_storage");
        assert(y);

        Tensor tx("x", Shape({1, 32}), DType::Float32, Device::cuda(0));
        Tensor tw("w", Shape({1, 32}), DType::Q8_0, Device::cuda(0));
        check_status(tx.allocate_cuda());
        check_status(tw.allocate_cuda_bytes(1));

        RuntimeContext ctx;
        check_status(ctx.bind(*x, &tx));
        check_status(ctx.bind(*w, &tw));
        check_status(ctx.allocate_intermediates(graph));

        KernelRegistry registry;
        register_cuda_kernels(registry);
        CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
        check_status(executor.compile(graph));
        auto st = executor.run(ctx);
        assert(!st.ok());
        assert(std::string(st.message()).find("too small") != std::string::npos);
    }

    std::cout << "  PASS test_cuda_executor_q8_0_rejects_invalid_inputs\n";
}

void test_cuda_executor_rope_custom_base() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({2, 8}), DType::Float32, Device::cuda(0));
    assert(x);
    auto y = gb.rope(*x, 2, 4, 500000.0, "rope");
    assert(y);
    auto out = gb.output(*y, "out");
    assert(out);
    check_status(g.validate());

    Tensor tx("x", Shape({2, 8}), DType::Float32, Device::cuda(0));
    check_status(tx.allocate_cuda());
    const std::vector<float> hx{0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7};
    check_cuda(cudaMemcpy(tx.data(), hx.data(), hx.size()*sizeof(float), cudaMemcpyHostToDevice));

    RuntimeContext ctx;
    check_status(ctx.bind(*x, &tx));
    check_status(ctx.allocate_intermediates(g));

    KernelRegistry registry;
    register_cuda_kernels(registry);
    CudaExecutor executor(std::make_shared<CudaBackend>(), registry);
    check_status(executor.compile(g));
    check_status(executor.run(ctx));
    sync_ok();

    Tensor* y_tensor = ctx.get(*y);
    assert(y_tensor != nullptr);
    assert(y_tensor->device().type == DeviceType::CUDA);
    std::cout << "  PASS test_cuda_executor_rope_custom_base\n";
}

void test_cuda_planned_memory_arena() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({64}), DType::Float32, Device::cuda(0));
    assert(x);
    auto y = gb.silu(*x, "y");
    assert(y);
    auto out = gb.output(*y, "out");
    assert(out);
    check_status(g.validate());

    Tensor tx("x", Shape({64}), DType::Float32, Device::cuda(0));
    check_status(tx.allocate_cuda());

    RuntimeContext ctx;
    check_status(ctx.bind(*x, &tx));
    {
        auto plan = ctx.allocate_intermediates_planned(g);
        assert(plan);
    }

    Tensor* y_tensor = ctx.get(*y);
    assert(y_tensor != nullptr);
    assert(y_tensor->device().type == DeviceType::CUDA);
    assert(y_tensor->is_allocated());
    std::cout << "  PASS test_cuda_planned_memory_arena\n";
}

int main() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count));
    assert(count > 0);
    check_cuda(cudaSetDevice(0));

    std::cout << "test_cuda_kernels:\n";
    test_cuda_elementwise();
    test_cuda_gemm_and_bias();
    test_cuda_bf16_weight_kernels();
    test_cuda_q8_0_weight_kernels();
    test_cuda_norm_embedding_rope_softmax_transpose();
    test_cuda_sdpa_gqa();
    test_cuda_paged_attention_decode_gqa();
    test_cuda_paged_attention_decode_batch_gqa();
    test_cuda_kv_cache_attention_decode_gqa();
    test_cuda_executor_linear_bias();
    test_cuda_executor_q8_0_linear_and_matmul();
    test_cuda_executor_q8_0_rejects_invalid_inputs();
    test_cuda_executor_rope_custom_base();
    test_cuda_planned_memory_arena();
    std::cout << "All tests passed!\n";
    return 0;
}
