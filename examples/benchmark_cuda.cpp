#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "minillm/runtime/kernels/cuda_kernels.h"

using namespace minillm;

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;
std::mt19937 rng(123);

void randf(std::vector<float>& v) {
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& x : v) x = d(rng);
}

void rand_id(std::vector<int>& v, int max) {
    std::uniform_int_distribution<int> d(0, max - 1);
    for (auto& x : v) x = d(rng);
}

bool ok_cuda(const char* what, cudaError_t err) {
    if (err == cudaSuccess) return true;
    std::cerr << what << ": " << cudaGetErrorString(err) << "\n";
    return false;
}

bool ok_status(const char* what, const Status& st) {
    if (st.ok()) return true;
    std::cerr << what << ": " << st.to_string() << "\n";
    return false;
}

bool sync_ok(const char* what) {
    return ok_cuda(what, cudaDeviceSynchronize());
}

size_t env_mib(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || (end && *end != '\0')) return fallback;
    return static_cast<size_t>(parsed);
}

std::string mib_string(size_t bytes) {
    double mib = static_cast<double>(bytes) / static_cast<double>(kMiB);
    return std::to_string(static_cast<unsigned long long>(mib + 0.5));
}

bool checked_mul(size_t a, size_t b, size_t* out, const char* label) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        std::cerr << label << ": size overflow\n";
        return false;
    }
    *out = a * b;
    return true;
}

bool product(std::initializer_list<size_t> dims, size_t* out, const char* label) {
    size_t total = 1;
    for (size_t dim : dims) {
        if (!checked_mul(total, dim, &total, label)) return false;
    }
    *out = total;
    return true;
}

template <typename T>
bool add_bytes(size_t* total, size_t count, const char* label) {
    size_t bytes = 0;
    if (!checked_mul(count, sizeof(T), &bytes, label)) return false;
    if (*total > std::numeric_limits<size_t>::max() - bytes) {
        std::cerr << label << ": total byte size overflow\n";
        return false;
    }
    *total += bytes;
    return true;
}

bool device_budget_ok(const char* name, size_t required_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!ok_cuda("cudaMemGetInfo", cudaMemGetInfo(&free_bytes, &total_bytes))) return false;

    const size_t reserve_bytes = env_mib("MINILLM_BENCH_GPU_RESERVE_MB", 512) * kMiB;
    const size_t max_device_bytes = env_mib("MINILLM_BENCH_MAX_DEVICE_MB", 3072) * kMiB;
    if (max_device_bytes != 0 && required_bytes > max_device_bytes) {
        std::cout << std::setw(20) << name << "SKIP requested=" << mib_string(required_bytes)
                  << " MiB cap=" << mib_string(max_device_bytes)
                  << " MiB (set MINILLM_BENCH_MAX_DEVICE_MB=0 to disable)\n";
        return false;
    }
    if (free_bytes <= reserve_bytes || required_bytes > free_bytes - reserve_bytes) {
        std::cout << std::setw(20) << name << "SKIP requested=" << mib_string(required_bytes)
                  << " MiB free=" << mib_string(free_bytes)
                  << " MiB reserve=" << mib_string(reserve_bytes) << " MiB\n";
        return false;
    }
    (void)total_bytes;
    return true;
}

template <typename T>
bool resize_host(std::vector<T>& v, size_t count, const char* label) {
    try {
        v.resize(count);
        return true;
    } catch (const std::bad_alloc&) {
        std::cerr << label << ": host allocation failed for " << count << " elements\n";
        return false;
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { reset(); }

    bool allocate(size_t count, const char* name) {
        reset();
        name_ = name;
        count_ = count;
        if (count == 0) return true;
        size_t bytes = 0;
        if (!checked_mul(count, sizeof(T), &bytes, name)) return false;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes);
        if (!ok_cuda((std::string("cudaMalloc ") + name).c_str(), err)) {
            ptr_ = nullptr;
            count_ = 0;
            return false;
        }
        return true;
    }

    bool copy_from(const std::vector<T>& host) const {
        if (host.size() != count_) {
            std::cerr << "cudaMemcpy " << name_ << ": host/device element count mismatch\n";
            return false;
        }
        if (count_ == 0) return true;
        size_t bytes = 0;
        if (!checked_mul(count_, sizeof(T), &bytes, name_.c_str())) return false;
        return ok_cuda((std::string("cudaMemcpy H2D ") + name_).c_str(),
                       cudaMemcpy(ptr_, host.data(), bytes, cudaMemcpyHostToDevice));
    }

    T* get() const { return ptr_; }

private:
    void reset() {
        if (!ptr_) return;
        cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree " << name_ << ": " << cudaGetErrorString(err) << "\n";
        }
        ptr_ = nullptr;
        count_ = 0;
    }

    T* ptr_{nullptr};
    size_t count_{0};
    std::string name_;
};

template <typename Fn>
bool time_cuda(const char* name, int iters, Fn&& fn, double* ms) {
    if (iters <= 0) {
        std::cerr << name << ": iters must be positive\n";
        return false;
    }
    if (!ok_status((std::string(name) + " warmup").c_str(), fn())) return false;
    if (!sync_ok((std::string(name) + " warmup sync").c_str())) return false;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!ok_status((std::string(name) + " launch").c_str(), fn())) return false;
    }
    if (!sync_ok((std::string(name) + " sync").c_str())) return false;
    auto t1 = std::chrono::steady_clock::now();
    *ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    return true;
}

} // namespace

void run() {
    if (!ok_cuda("cudaSetDevice", cudaSetDevice(0))) return;
    std::cout << std::left;

    // ============= GEMM =============
    {
        const int M = 4, N = 4096, K = 4096, iters = 50;
        size_t a_count = 0, b_count = 0, c_count = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(M), static_cast<size_t>(K)}, &a_count, "sgemm_nt A") ||
            !product({static_cast<size_t>(N), static_cast<size_t>(K)}, &b_count, "sgemm_nt B") ||
            !product({static_cast<size_t>(M), static_cast<size_t>(N)}, &c_count, "sgemm_nt C") ||
            !add_bytes<float>(&device_bytes, a_count, "sgemm_nt A") ||
            !add_bytes<float>(&device_bytes, b_count, "sgemm_nt B") ||
            !add_bytes<float>(&device_bytes, c_count, "sgemm_nt C")) return;
        if (device_budget_ok("sgemm_nt", device_bytes)) {
            std::vector<float> hA, hB, hC;
            if (!resize_host(hA, a_count, "sgemm_nt A") ||
                !resize_host(hB, b_count, "sgemm_nt B") ||
                !resize_host(hC, c_count, "sgemm_nt C")) return;
            randf(hA); randf(hB);
            DeviceBuffer<float> dA, dB, dC;
            if (!dA.allocate(a_count, "sgemm_nt A") || !dB.allocate(b_count, "sgemm_nt B") ||
                !dC.allocate(c_count, "sgemm_nt C") || !dA.copy_from(hA) || !dB.copy_from(hB)) return;
            double ms = 0.0;
            if (!time_cuda("sgemm_nt", iters, [&] {
                    return cuda::sgemm_nt(dA.get(), dB.get(), dC.get(), M, N, K);
                }, &ms)) return;
            std::cout << std::setw(20) << "sgemm_nt" << std::setw(18) << "[4,4096]x[4096,4096]"
                      << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms
                      << " gflops=" << std::setprecision(1) << 2.0 * M * N * K / ms / 1e6 << "\n";
        }
    }
    {
        const int M = 1, N = 4096, K = 4096, iters = 100;
        size_t a_count = 0, b_count = 0, c_count = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(M), static_cast<size_t>(K)}, &a_count, "decode A") ||
            !product({static_cast<size_t>(N), static_cast<size_t>(K)}, &b_count, "decode B") ||
            !product({static_cast<size_t>(M), static_cast<size_t>(N)}, &c_count, "decode C") ||
            !add_bytes<float>(&device_bytes, a_count, "decode A") ||
            !add_bytes<float>(&device_bytes, b_count, "decode B") ||
            !add_bytes<float>(&device_bytes, c_count, "decode C")) return;
        if (device_budget_ok("sgemm_nt decode", device_bytes)) {
            std::vector<float> hA, hB, hC;
            if (!resize_host(hA, a_count, "decode A") || !resize_host(hB, b_count, "decode B") ||
                !resize_host(hC, c_count, "decode C")) return;
            randf(hA); randf(hB);
            DeviceBuffer<float> dA, dB, dC;
            if (!dA.allocate(a_count, "decode A") || !dB.allocate(b_count, "decode B") ||
                !dC.allocate(c_count, "decode C") || !dA.copy_from(hA) || !dB.copy_from(hB)) return;
            double ms = 0.0;
            if (!time_cuda("sgemm_nt decode", iters, [&] {
                    return cuda::sgemm_nt(dA.get(), dB.get(), dC.get(), M, N, K);
                }, &ms)) return;
            std::cout << std::setw(20) << "sgemm_nt (decode)" << std::setw(18) << "[1,4096]x[4096,4096]"
                      << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms
                      << " gflops=" << std::setprecision(1) << 2.0 * M * N * K / ms / 1e6 << "\n";
        }
    }

    // ============= Element-wise ops =============
    auto bench_ew = [&](const char* name, size_t n, int iters, auto&& fn) {
        size_t device_bytes = 0;
        if (!add_bytes<float>(&device_bytes, n, name) ||
            !add_bytes<float>(&device_bytes, n, name) ||
            !add_bytes<float>(&device_bytes, n, name)) return false;
        if (!device_budget_ok(name, device_bytes)) return true;
        std::vector<float> hA, hB, hC;
        if (!resize_host(hA, n, name) || !resize_host(hB, n, name) || !resize_host(hC, n, name)) return false;
        randf(hA); randf(hB);
        DeviceBuffer<float> dA, dB, dC;
        if (!dA.allocate(n, (std::string(name) + " A").c_str()) ||
            !dB.allocate(n, (std::string(name) + " B").c_str()) ||
            !dC.allocate(n, (std::string(name) + " C").c_str()) ||
            !dA.copy_from(hA) || !dB.copy_from(hB)) return false;
        double ms = 0.0;
        if (!time_cuda(name, iters, [&] { return fn(dA.get(), dB.get(), dC.get(), static_cast<int>(n)); }, &ms)) return false;
        std::cout << std::setw(20) << name << std::setw(18) << ("n=" + std::to_string(n))
                  << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        return true;
    };

    const size_t n_ew = 4096ull * 4096ull;
    if (!bench_ew("add", n_ew, 200, [](auto* a, auto* b, auto* c, int n) { return cuda::add(a, b, c, n); })) return;
    if (!bench_ew("mul", n_ew, 200, [](auto* a, auto* b, auto* c, int n) { return cuda::mul(a, b, c, n); })) return;
    if (!bench_ew("silu", n_ew, 200, [](auto* a, auto*, auto* c, int n) { return cuda::silu(a, c, n); })) return;
    if (!bench_ew("fused_silu_mul", n_ew, 200, [](auto* a, auto* b, auto* c, int n) { return cuda::fused_silu_mul(a, b, c, n); })) return;

    // ============= RMSNorm =============
    {
        const int rows = 4096, cols = 1024, iters = 200;
        size_t n = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(rows), static_cast<size_t>(cols)}, &n, "rmsnorm") ||
            !add_bytes<float>(&device_bytes, n, "rmsnorm X") ||
            !add_bytes<float>(&device_bytes, static_cast<size_t>(cols), "rmsnorm gamma") ||
            !add_bytes<float>(&device_bytes, n, "rmsnorm Y")) return;
        if (device_budget_ok("rmsnorm", device_bytes)) {
            std::vector<float> hX, hG, hY;
            if (!resize_host(hX, n, "rmsnorm X") || !resize_host(hG, cols, "rmsnorm gamma") ||
                !resize_host(hY, n, "rmsnorm Y")) return;
            randf(hX); randf(hG);
            DeviceBuffer<float> dX, dG, dY;
            if (!dX.allocate(n, "rmsnorm X") || !dG.allocate(cols, "rmsnorm gamma") ||
                !dY.allocate(n, "rmsnorm Y") || !dX.copy_from(hX) || !dG.copy_from(hG)) return;
            double ms = 0.0;
            if (!time_cuda("rmsnorm", iters, [&] {
                    return cuda::rmsnorm(dX.get(), dG.get(), dY.get(), rows, cols, 1e-6f);
                }, &ms)) return;
            std::cout << std::setw(20) << "rmsnorm" << std::setw(18)
                      << ("[" + std::to_string(rows) + "," + std::to_string(cols) + "]")
                      << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    // ============= Embedding =============
    {
        const int seq = 256, vocab = 151936, hidden = 1024, iters = 50;
        size_t weight_count = 0, out_count = 0, ids_count = static_cast<size_t>(seq), device_bytes = 0;
        if (!product({static_cast<size_t>(vocab), static_cast<size_t>(hidden)}, &weight_count, "embedding weight") ||
            !product({static_cast<size_t>(seq), static_cast<size_t>(hidden)}, &out_count, "embedding out") ||
            !add_bytes<float>(&device_bytes, weight_count, "embedding weight") ||
            !add_bytes<int>(&device_bytes, ids_count, "embedding ids") ||
            !add_bytes<float>(&device_bytes, out_count, "embedding out")) return;
        if (device_budget_ok("embedding", device_bytes)) {
            std::vector<float> hW, hO;
            std::vector<int> hIDs;
            if (!resize_host(hW, weight_count, "embedding weight") ||
                !resize_host(hIDs, ids_count, "embedding ids") ||
                !resize_host(hO, out_count, "embedding out")) return;
            randf(hW); rand_id(hIDs, vocab);
            DeviceBuffer<float> dW, dO;
            DeviceBuffer<int> dIDs;
            if (!dW.allocate(weight_count, "embedding weight") || !dIDs.allocate(ids_count, "embedding ids") ||
                !dO.allocate(out_count, "embedding out") || !dW.copy_from(hW) || !dIDs.copy_from(hIDs)) return;
            double ms = 0.0;
            if (!time_cuda("embedding", iters, [&] {
                    return cuda::embedding(dW.get(), dIDs.get(), dO.get(), seq, vocab, hidden);
                }, &ms)) return;
            std::cout << std::setw(20) << "embedding" << std::setw(18)
                      << ("seq=" + std::to_string(seq) + " vocab=" + std::to_string(vocab))
                      << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    // ============= RoPE =============
    {
        const int tokens = 256, nh = 16, hd = 128, iters = 100;
        size_t n = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(tokens), static_cast<size_t>(nh), static_cast<size_t>(hd)}, &n, "rope") ||
            !add_bytes<float>(&device_bytes, n, "rope X") || !add_bytes<float>(&device_bytes, n, "rope O")) return;
        if (device_budget_ok("apply_rope", device_bytes)) {
            std::vector<float> hX, hO;
            if (!resize_host(hX, n, "rope X") || !resize_host(hO, n, "rope O")) return;
            randf(hX);
            DeviceBuffer<float> dX, dO;
            if (!dX.allocate(n, "rope X") || !dO.allocate(n, "rope O") || !dX.copy_from(hX)) return;
            double ms = 0.0;
            if (!time_cuda("apply_rope", iters, [&] {
                    return cuda::apply_rope(dX.get(), dO.get(), tokens, nh, hd, 10000.0f, 0);
                }, &ms)) return;
            std::cout << std::setw(20) << "apply_rope" << std::setw(18)
                      << ("tok=" + std::to_string(tokens) + " hd=" + std::to_string(hd))
                      << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    // ============= Softmax =============
    {
        const int rows = 4096, cols = 151936, iters = 20;
        size_t n = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(rows), static_cast<size_t>(cols)}, &n, "softmax") ||
            !add_bytes<float>(&device_bytes, n, "softmax X") ||
            !add_bytes<float>(&device_bytes, n, "softmax O")) return;
        if (device_budget_ok("softmax", device_bytes)) {
            std::vector<float> hX, hO;
            if (!resize_host(hX, n, "softmax X") || !resize_host(hO, n, "softmax O")) return;
            randf(hX);
            DeviceBuffer<float> dX, dO;
            if (!dX.allocate(n, "softmax X") || !dO.allocate(n, "softmax O") || !dX.copy_from(hX)) return;
            double ms = 0.0;
            if (!time_cuda("softmax", iters, [&] { return cuda::softmax(dX.get(), dO.get(), rows, cols); }, &ms)) return;
            std::cout << std::setw(20) << "softmax" << std::setw(18)
                      << ("[" + std::to_string(rows) + "," + std::to_string(cols) + "]")
                      << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms << "\n";
        }
    }

    // ============= SDPA =============
    {
        const int b = 1, seq = 256, nh = 16, nkv = 8, hd = 128, iters = 20;
        size_t q_count = 0, kv_count = 0, o_count = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(b), static_cast<size_t>(seq), static_cast<size_t>(nh), static_cast<size_t>(hd)}, &q_count, "sdpa Q") ||
            !product({static_cast<size_t>(b), static_cast<size_t>(seq), static_cast<size_t>(nkv), static_cast<size_t>(hd)}, &kv_count, "sdpa KV") ||
            !product({static_cast<size_t>(b), static_cast<size_t>(seq), static_cast<size_t>(nh), static_cast<size_t>(hd)}, &o_count, "sdpa O") ||
            !add_bytes<float>(&device_bytes, q_count, "sdpa Q") || !add_bytes<float>(&device_bytes, kv_count, "sdpa K") ||
            !add_bytes<float>(&device_bytes, kv_count, "sdpa V") || !add_bytes<float>(&device_bytes, o_count, "sdpa O")) return;
        if (device_budget_ok("sdpa", device_bytes)) {
            std::vector<float> hQ, hK, hV, hO;
            if (!resize_host(hQ, q_count, "sdpa Q") || !resize_host(hK, kv_count, "sdpa K") ||
                !resize_host(hV, kv_count, "sdpa V") || !resize_host(hO, o_count, "sdpa O")) return;
            randf(hQ); randf(hK); randf(hV);
            DeviceBuffer<float> dQ, dK, dV, dO;
            if (!dQ.allocate(q_count, "sdpa Q") || !dK.allocate(kv_count, "sdpa K") ||
                !dV.allocate(kv_count, "sdpa V") || !dO.allocate(o_count, "sdpa O") ||
                !dQ.copy_from(hQ) || !dK.copy_from(hK) || !dV.copy_from(hV)) return;
            double ms = 0.0;
            if (!time_cuda("sdpa", iters, [&] { return cuda::sdpa(dQ.get(), dK.get(), dV.get(), dO.get(), b, seq, nh, nkv, hd, true); }, &ms)) return;
            std::cout << std::setw(20) << "sdpa" << std::setw(18) << ("seq=" + std::to_string(seq) + " nh=16")
                      << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms << "\n";
        }
    }

    // ============= KV cache attention decode =============
    {
        const int kvlen = 128, nh = 16, nkv = 8, hd = 128, iters = 200;
        size_t q_count = 0, kv_count = 0, o_count = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(nh), static_cast<size_t>(hd)}, &q_count, "kv attn Q") ||
            !product({static_cast<size_t>(kvlen), static_cast<size_t>(nkv), static_cast<size_t>(hd)}, &kv_count, "kv attn KV") ||
            !product({static_cast<size_t>(nh), static_cast<size_t>(hd)}, &o_count, "kv attn O") ||
            !add_bytes<float>(&device_bytes, q_count, "kv attn Q") || !add_bytes<float>(&device_bytes, kv_count, "kv attn K") ||
            !add_bytes<float>(&device_bytes, kv_count, "kv attn V") || !add_bytes<float>(&device_bytes, o_count, "kv attn O")) return;
        if (device_budget_ok("kv_cache_attn", device_bytes)) {
            std::vector<float> hQ, hK, hV, hO;
            if (!resize_host(hQ, q_count, "kv attn Q") || !resize_host(hK, kv_count, "kv attn K") ||
                !resize_host(hV, kv_count, "kv attn V") || !resize_host(hO, o_count, "kv attn O")) return;
            randf(hQ); randf(hK); randf(hV);
            DeviceBuffer<float> dQ, dK, dV, dO;
            if (!dQ.allocate(q_count, "kv attn Q") || !dK.allocate(kv_count, "kv attn K") ||
                !dV.allocate(kv_count, "kv attn V") || !dO.allocate(o_count, "kv attn O") ||
                !dQ.copy_from(hQ) || !dK.copy_from(hK) || !dV.copy_from(hV)) return;
            double ms = 0.0;
            if (!time_cuda("kv_cache_attn", iters, [&] {
                    return cuda::kv_cache_attention_decode(dQ.get(), dK.get(), dV.get(), dO.get(), kvlen, nh, nkv, hd);
                }, &ms)) return;
            std::cout << std::setw(20) << "kv_cache_attn" << std::setw(18)
                      << ("kv=" + std::to_string(kvlen) + " nh=16") << " us=" << std::setw(10)
                      << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    // ============= Paged attention decode =============
    {
        const int kvlen = 128, nh = 16, nkv = 8, hd = 128, bs = 16, max_blocks = 64;
        const int nb = (kvlen + bs - 1) / bs;
        size_t q_count = 0, o_count = 0, cache_count = 0, bt_count = static_cast<size_t>(nb), device_bytes = 0;
        if (!product({static_cast<size_t>(nh), static_cast<size_t>(hd)}, &q_count, "paged attn Q") ||
            !product({static_cast<size_t>(nh), static_cast<size_t>(hd)}, &o_count, "paged attn O") ||
            !product({static_cast<size_t>(max_blocks), static_cast<size_t>(bs), static_cast<size_t>(nkv), static_cast<size_t>(hd)}, &cache_count, "paged attn cache") ||
            !add_bytes<float>(&device_bytes, q_count, "paged attn Q") || !add_bytes<float>(&device_bytes, cache_count, "paged attn K") ||
            !add_bytes<float>(&device_bytes, cache_count, "paged attn V") || !add_bytes<float>(&device_bytes, o_count, "paged attn O") ||
            !add_bytes<int>(&device_bytes, bt_count, "paged attn block table")) return;
        if (device_budget_ok("paged_attn_decode", device_bytes)) {
            std::vector<float> hQ, hK, hV, hO;
            std::vector<int> hBT;
            if (!resize_host(hQ, q_count, "paged attn Q") || !resize_host(hK, cache_count, "paged attn K") ||
                !resize_host(hV, cache_count, "paged attn V") || !resize_host(hO, o_count, "paged attn O") ||
                !resize_host(hBT, bt_count, "paged attn block table")) return;
            for (int i = 0; i < nb; ++i) hBT[static_cast<size_t>(i)] = i;
            randf(hQ); randf(hK); randf(hV);
            DeviceBuffer<float> dQ, dK, dV, dO;
            DeviceBuffer<int> dBT;
            if (!dQ.allocate(q_count, "paged attn Q") || !dK.allocate(cache_count, "paged attn K") ||
                !dV.allocate(cache_count, "paged attn V") || !dO.allocate(o_count, "paged attn O") ||
                !dBT.allocate(bt_count, "paged attn block table") || !dQ.copy_from(hQ) ||
                !dK.copy_from(hK) || !dV.copy_from(hV) || !dBT.copy_from(hBT)) return;
            const int iters = 200;
            double ms = 0.0;
            if (!time_cuda("paged_attn_decode", iters, [&] {
                    return cuda::paged_attention_decode(dQ.get(), dK.get(), dV.get(), dBT.get(), dO.get(), kvlen, nh, nkv, hd, bs);
                }, &ms)) return;
            std::cout << std::setw(20) << "paged_attn_decode" << std::setw(18)
                      << ("kv=" + std::to_string(kvlen) + " nh=16") << " us=" << std::setw(10)
                      << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    // ============= Transpose =============
    {
        const int a = 64, b = 128, iters = 1000;
        size_t n = 0, device_bytes = 0;
        if (!product({static_cast<size_t>(a), static_cast<size_t>(b)}, &n, "transpose") ||
            !add_bytes<float>(&device_bytes, n, "transpose X") || !add_bytes<float>(&device_bytes, n, "transpose O")) return;
        if (device_budget_ok("transpose", device_bytes)) {
            std::vector<float> hX, hO;
            if (!resize_host(hX, n, "transpose X") || !resize_host(hO, n, "transpose O")) return;
            randf(hX);
            DeviceBuffer<float> dX, dO;
            if (!dX.allocate(n, "transpose X") || !dO.allocate(n, "transpose O") || !dX.copy_from(hX)) return;
            int64_t dims[2] = {a, b};
            double ms = 0.0;
            if (!time_cuda("transpose", iters, [&] { return cuda::transpose(dX.get(), dO.get(), dims, 2, 0, 1); }, &ms)) return;
            std::cout << std::setw(20) << "transpose" << std::setw(18)
                      << ("[" + std::to_string(a) + "," + std::to_string(b) + "]") << " us=" << std::setw(10)
                      << std::fixed << std::setprecision(2) << ms * 1000 << "\n";
        }
    }

    std::cout << "\nAll CUDA operator benchmarks complete.\n";
}

int main() {
    run();
    return 0;
}
