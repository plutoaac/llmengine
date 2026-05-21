#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <memory>
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

    check_status(cuda::sgemm_nt(dA.get(), dBt.get(), dC.get(), 2, 2, 3));
    sync_ok();
    C = dC.copy_to_host();
    const std::vector<float> expected_nt{50, 68, 122, 167};
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

    std::cout << "  PASS test_cuda_gemm_and_bias\n";
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

    std::cout << "  PASS test_cuda_sdpa_gqa\n";
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

int main() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count));
    assert(count > 0);
    check_cuda(cudaSetDevice(0));

    std::cout << "test_cuda_kernels:\n";
    test_cuda_elementwise();
    test_cuda_gemm_and_bias();
    test_cuda_norm_embedding_rope_softmax_transpose();
    test_cuda_sdpa_gqa();
    test_cuda_executor_linear_bias();
    std::cout << "All tests passed!\n";
    return 0;
}
