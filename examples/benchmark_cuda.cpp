#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "minillm/runtime/cuda_kernels.h"

static void fill_random(std::vector<float>& data) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : data) v = dist(rng);
}

static bool chk(const char* what, cudaError_t err) {
    if (err == cudaSuccess) return true;
    std::cerr << what << ": " << cudaGetErrorString(err) << "\n";
    return false;
}

struct Bench { const char* name; double ms; double gflops; };
static void print(const Bench& b) {
    std::cout << b.name << " avg_ms=" << b.ms << " gflops=" << b.gflops << "\n";
}

int main() {
    if (!chk("cudaSetDevice", cudaSetDevice(0))) return 1;

    // GEMM: C[M,N] = A[M,K] x B^T[K,N]  (sgemm_nt)
    {
        int M = 4, N = 4096, K = 4096, iters = 50;
        std::vector<float> hA(M*K), hB(N*K), hC(M*N);
        fill_random(hA); fill_random(hB);

        float *dA, *dB, *dC;
        chk("malloc A", cudaMalloc(&dA, hA.size()*4));
        chk("malloc B", cudaMalloc(&dB, hB.size()*4));
        chk("malloc C", cudaMalloc(&dC, hC.size()*4));
        cudaMemcpy(dA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice);

        minillm::cuda::sgemm_nt(dA, dB, dC, M, N, K); // warmup
        cudaDeviceSynchronize();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) minillm::cuda::sgemm_nt(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;
        double gflops = 2.0 * M * N * K / ms / 1e6;
        print({"cuda sgemm_nt (M=4, NxK=4096)", ms, gflops});

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // M=1 decode-style
    {
        int M = 1, N = 4096, K = 4096, iters = 100;
        std::vector<float> hA(M*K), hB(N*K), hC(M*N);
        fill_random(hA); fill_random(hB);
        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size()*4); cudaMalloc(&dB, hB.size()*4); cudaMalloc(&dC, hC.size()*4);
        cudaMemcpy(dA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice);
        minillm::cuda::sgemm_nt(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) minillm::cuda::sgemm_nt(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;
        double gflops = 2.0 * M * N * K / ms / 1e6;
        print({"cuda sgemm_nt (M=1, NxK=4096)", ms, gflops});
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // Attention (SDPA)
    {
        int batch=1, seq=256, nh=16, nkv=8, hd=128;
        std::vector<float> hQ(batch*seq*nh*hd), hK(batch*seq*nkv*hd), hV(batch*seq*nkv*hd), hO(batch*seq*nh*hd);
        fill_random(hQ); fill_random(hK); fill_random(hV);
        float *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, hQ.size()*4); cudaMalloc(&dK, hK.size()*4);
        cudaMalloc(&dV, hV.size()*4); cudaMalloc(&dO, hO.size()*4);
        cudaMemcpy(dQ, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dK, hK.data(), hK.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hV.data(), hV.size()*4, cudaMemcpyHostToDevice);

        minillm::cuda::sdpa(dQ, dK, dV, dO, batch, seq, nh, nkv, hd, true);
        cudaDeviceSynchronize();

        int iters = 20;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) minillm::cuda::sdpa(dQ, dK, dV, dO, batch, seq, nh, nkv, hd, true);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;
        print({"cuda sdpa (seq=256, 16 heads)", ms, 0});
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    std::cout << "\nAll CUDA benchmarks complete.\n";
    return 0;
}
