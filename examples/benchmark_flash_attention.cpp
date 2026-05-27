#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "minillm/runtime/kernels/cpu_kernels.h"

using namespace minillm;

namespace {

void fill_deterministic(std::vector<float>& data) {
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        int raw = (i * 17 + 11) % 37;
        data[static_cast<size_t>(i)] = (static_cast<float>(raw) - 18.0f) / 37.0f;
    }
}

double max_relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double denom = std::max(1e-6, static_cast<double>(std::abs(b[i])));
        double err = static_cast<double>(std::abs(a[i] - b[i])) / denom;
        max_err = std::max(max_err, err);
    }
    return max_err;
}

template <typename Fn>
double bench_seconds(Fn&& fn, int iters) {
    fn();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void run_case(int seq_len, int head_dim, int heads, int iters) {
    const int kv_len = seq_len;
    const size_t q_bytes = static_cast<size_t>(heads) * seq_len * head_dim;
    const size_t kv_bytes = static_cast<size_t>(heads) * kv_len * head_dim;

    std::vector<float> q(q_bytes);
    std::vector<float> k(kv_bytes);
    std::vector<float> v(kv_bytes);
    std::vector<float> naive(q_bytes);
    std::vector<float> flash(q_bytes);
    fill_deterministic(q);
    fill_deterministic(k);
    fill_deterministic(v);

    auto naive_fn = [&] {
        cpu::sdpa(q.data(), k.data(), v.data(), naive.data(),
                  heads, seq_len, kv_len, head_dim, true);
    };
    auto flash_fn = [&] {
        cpu::flash_sdpa(q.data(), k.data(), v.data(), flash.data(),
                        heads, seq_len, kv_len, head_dim, true);
    };

    naive_fn();
    flash_fn();
    const double rel_err = max_relative_error(flash, naive);

    const double naive_s = bench_seconds(naive_fn, iters);
    const double flash_s = bench_seconds(flash_fn, iters);
    const double naive_ms = naive_s * 1000.0 / static_cast<double>(iters);
    const double flash_ms = flash_s * 1000.0 / static_cast<double>(iters);
    const double speedup = naive_ms / flash_ms;

    std::cout << std::setw(7) << seq_len
              << std::setw(9) << head_dim
              << std::setw(7) << heads
              << std::setw(13) << std::fixed << std::setprecision(3) << naive_ms
              << std::setw(13) << std::fixed << std::setprecision(3) << flash_ms
              << std::setw(10) << std::fixed << std::setprecision(2) << speedup
              << std::setw(12) << std::scientific << std::setprecision(2) << rel_err
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 2;
    int heads = argc > 2 ? std::atoi(argv[2]) : 8;
    if (iters <= 0 || heads <= 0) {
        std::cerr << "Usage: " << argv[0] << " [iters] [heads]\n";
        return 1;
    }

    std::cout << "MiniLLM CPU FlashAttention benchmark\n";
    std::cout << "causal=true, kv_len=seq_len, deterministic FP32 inputs\n";
    std::cout << std::setw(7) << "seq"
              << std::setw(9) << "head_dim"
              << std::setw(7) << "heads"
              << std::setw(13) << "sdpa_ms"
              << std::setw(13) << "flash_ms"
              << std::setw(10) << "speedup"
              << std::setw(12) << "max_rel_err"
              << "\n";

    for (int seq : {128, 256, 512}) {
        for (int head_dim : {64, 128}) {
            run_case(seq, head_dim, heads, iters);
        }
    }

    return 0;
}
