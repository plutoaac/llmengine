#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "minillm/runtime/cpu_kernels.h"

using namespace minillm;

static void fill_random(std::vector<float>& data) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : data) {
        v = dist(rng);
    }
}

static double benchmark_sgemm_nt(int M, int N, int K, int iters) {
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(N) * K);
    std::vector<float> C(static_cast<size_t>(M) * N);
    fill_random(A);
    fill_random(B);

    cpu::sgemm_nt(A.data(), B.data(), C.data(), M, N, K);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu::sgemm_nt(A.data(), B.data(), C.data(), M, N, K);
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    volatile float sink = C[static_cast<size_t>(M) * N / 2];
    (void)sink;
    return elapsed.count();
}

static double benchmark_sgemm(int M, int N, int K, int iters) {
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N);
    fill_random(A);
    fill_random(B);

    cpu::sgemm(A.data(), B.data(), C.data(), M, N, K);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu::sgemm(A.data(), B.data(), C.data(), M, N, K);
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    volatile float sink = C[static_cast<size_t>(M) * N / 2];
    (void)sink;
    return elapsed.count();
}

static void print_result(const char* name, int M, int N, int K, int iters, double seconds) {
    const double flops = 2.0 * M * N * K * iters;
    const double gflops = flops / seconds / 1e9;
    const double ms = seconds * 1000.0 / iters;

    std::cout << std::left << std::setw(12) << name
              << " shape=[" << M << "," << N << "," << K << "]"
              << " iters=" << iters
              << " avg_ms=" << std::fixed << std::setprecision(4) << ms
              << " gflops=" << std::fixed << std::setprecision(2) << gflops
              << "\n";
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 1;
    int N = argc > 2 ? std::atoi(argv[2]) : 2048;
    int K = argc > 3 ? std::atoi(argv[3]) : 2048;
    int iters = argc > 4 ? std::atoi(argv[4]) : 20;

    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0) {
        std::cerr << "Usage: " << argv[0] << " [M] [N] [K] [iters]\n";
        return 1;
    }

    std::cout << "MiniLLM CPU GEMM benchmark\n";
    std::cout << "sgemm_nt matches Linear: A[M,K] x W[N,K]^T -> C[M,N]\n";

    double nt_seconds = benchmark_sgemm_nt(M, N, K, iters);
    print_result("sgemm_nt", M, N, K, iters, nt_seconds);

    double seconds = benchmark_sgemm(M, N, K, iters);
    print_result("sgemm", M, N, K, iters, seconds);

    return 0;
}
