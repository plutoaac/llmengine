#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "minillm/runtime/cuda_kernels.h"

using namespace minillm;

static std::mt19937 rng(123);

static void randf(std::vector<float>& v) {
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& x : v) x = d(rng);
}
static void rand_id(std::vector<int>& v, int max) {
    std::uniform_int_distribution<int> d(0, max - 1);
    for (auto& x : v) x = d(rng);
}

static bool ok(const char* s, cudaError_t e) {
    if (e == cudaSuccess) return true;
    std::cerr << s << ": " << cudaGetErrorString(e) << "\n";
    return false;
}

struct Row { const char* name; const char* shape; double ms; double gflops; };

void run() {
    if (!ok("cudaSetDevice", cudaSetDevice(0))) return;
    std::cout << std::left;

    // ============= GEMM =============
    {
        int M = 4, N = 4096, K = 4096, iters = 50;
        std::vector<float> hA(M*K), hB(N*K), hC(M*N); randf(hA); randf(hB);
        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size()*4); cudaMalloc(&dB, hB.size()*4); cudaMalloc(&dC, hC.size()*4);
        cudaMemcpy(dA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice);
        cuda::sgemm_nt(dA, dB, dC, M, N, K); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::sgemm_nt(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "sgemm_nt" << std::setw(18) << "[4,4096]x[4096,4096]"
                  << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms
                  << " gflops=" << std::setprecision(1) << 2.0*M*N*K/ms/1e6 << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    {
        int M = 1, N = 4096, K = 4096, iters = 100;
        std::vector<float> hA(M*K), hB(N*K), hC(M*N); randf(hA); randf(hB);
        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size()*4); cudaMalloc(&dB, hB.size()*4); cudaMalloc(&dC, hC.size()*4);
        cudaMemcpy(dA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice);
        cuda::sgemm_nt(dA, dB, dC, M, N, K); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::sgemm_nt(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "sgemm_nt (decode)" << std::setw(18) << "[1,4096]x[4096,4096]"
                  << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms
                  << " gflops=" << std::setprecision(1) << 2.0*M*N*K/ms/1e6 << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ============= Element-wise ops =============
    auto bench_ew = [&](const char* name, int n, int iters, auto fn) {
        std::vector<float> hA(n), hB(n), hC(n); randf(hA); randf(hB);
        float *dA, *dB, *dC;
        cudaMalloc(&dA, n*4); cudaMalloc(&dB, n*4); cudaMalloc(&dC, n*4);
        cudaMemcpy(dA, hA.data(), n*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), n*4, cudaMemcpyHostToDevice);
        fn(dA, dB, dC, n); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn(dA, dB, dC, n);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << name << std::setw(18) << ("n="+std::to_string(n))
                  << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    };

    int n_ew = 4096 * 4096;
    bench_ew("add", n_ew, 200, [](auto* a,auto* b,auto* c,int n){ cuda::add(a,b,c,n); });
    bench_ew("mul", n_ew, 200, [](auto* a,auto* b,auto* c,int n){ cuda::mul(a,b,c,n); });
    bench_ew("silu", n_ew, 200, [](auto* a,auto*,auto* c,int n){ cuda::silu(a,c,n); });
    bench_ew("fused_silu_mul", n_ew, 200, [](auto* a,auto* b,auto* c,int n){ cuda::fused_silu_mul(a,b,c,n); });

    // ============= RMSNorm =============
    {
        int rows = 4096, cols = 1024, iters = 200;
        int n = rows * cols;
        std::vector<float> hX(n), hG(cols), hY(n); randf(hX); randf(hG);
        float *dX, *dG, *dY;
        cudaMalloc(&dX, n*4); cudaMalloc(&dG, cols*4); cudaMalloc(&dY, n*4);
        cudaMemcpy(dX, hX.data(), n*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dG, hG.data(), cols*4, cudaMemcpyHostToDevice);
        cuda::rmsnorm(dX, dG, dY, rows, cols, 1e-6f); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::rmsnorm(dX, dG, dY, rows, cols, 1e-6f);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "rmsnorm" << std::setw(18)
                  << ("["+std::to_string(rows)+","+std::to_string(cols)+"]")
                  << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dX); cudaFree(dG); cudaFree(dY);
    }

    // ============= Embedding =============
    {
        int seq = 256, vocab = 151936, hidden = 1024, iters = 50;
        std::vector<float> hW(vocab*hidden); randf(hW);
        std::vector<int> hIDs(seq); rand_id(hIDs, vocab);
        std::vector<float> hO(seq*hidden);
        float *dW, *dO; int *dIDs;
        cudaMalloc(&dW, hW.size()*4); cudaMalloc(&dIDs, hIDs.size()*4); cudaMalloc(&dO, hO.size()*4);
        cudaMemcpy(dW, hW.data(), hW.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dIDs, hIDs.data(), hIDs.size()*4, cudaMemcpyHostToDevice);
        cuda::embedding(dW, dIDs, dO, seq, hidden, vocab); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::embedding(dW, dIDs, dO, seq, hidden, vocab);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "embedding" << std::setw(18)
                  << ("seq="+std::to_string(seq)+" vocab="+std::to_string(vocab))
                  << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dW); cudaFree(dIDs); cudaFree(dO);
    }

    // ============= RoPE =============
    {
        int tokens = 256, nh = 16, hd = 128, iters = 100;
        int n = tokens * nh * hd;
        std::vector<float> hX(n), hO(n); randf(hX);
        float *dX, *dO;
        cudaMalloc(&dX, n*4); cudaMalloc(&dO, n*4);
        cudaMemcpy(dX, hX.data(), n*4, cudaMemcpyHostToDevice);
        cuda::apply_rope(dX, dO, tokens, nh, hd, 10000.0f, 0); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::apply_rope(dX, dO, tokens, nh, hd, 10000.0f, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "apply_rope" << std::setw(18)
                  << ("tok="+std::to_string(tokens)+" hd="+std::to_string(hd))
                  << " us=" << std::setw(10) << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dX); cudaFree(dO);
    }

    // ============= Softmax =============
    {
        int rows = 4096, cols = 151936, iters = 20;
        int n = rows * cols;
        std::vector<float> hX(n), hO(n); randf(hX);
        float *dX, *dO;
        cudaMalloc(&dX, n*4); cudaMalloc(&dO, n*4);
        cudaMemcpy(dX, hX.data(), n*4, cudaMemcpyHostToDevice);
        cuda::softmax(dX, dO, rows, cols); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::softmax(dX, dO, rows, cols);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "softmax" << std::setw(18)
                  << ("["+std::to_string(rows)+","+std::to_string(cols)+"]")
                  << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms << "\n";
        cudaFree(dX); cudaFree(dO);
    }

    // ============= SDPA =============
    {
        int b=1, seq=256, nh=16, nkv=8, hd=128, iters=20;
        std::vector<float> hQ(b*seq*nh*hd), hK(b*seq*nkv*hd), hV(b*seq*nkv*hd), hO(b*seq*nh*hd);
        randf(hQ); randf(hK); randf(hV);
        float *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, hQ.size()*4); cudaMalloc(&dK, hK.size()*4);
        cudaMalloc(&dV, hV.size()*4); cudaMalloc(&dO, hO.size()*4);
        cudaMemcpy(dQ, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dK, hK.data(), hK.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hV.data(), hV.size()*4, cudaMemcpyHostToDevice);
        cuda::sdpa(dQ, dK, dV, dO, b, seq, nh, nkv, hd, true); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::sdpa(dQ, dK, dV, dO, b, seq, nh, nkv, hd, true);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "sdpa" << std::setw(18) << ("seq="+std::to_string(seq)+" nh=16")
                  << " ms=" << std::setw(10) << std::fixed << std::setprecision(4) << ms << "\n";
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    // ============= KV cache attention decode =============
    {
        int kvlen=128, nh=16, nkv=8, hd=128, iters=200;
        std::vector<float> hQ(nh*hd), hK(kvlen*nkv*hd), hV(kvlen*nkv*hd), hO(nh*hd);
        randf(hQ); randf(hK); randf(hV);
        float *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, hQ.size()*4); cudaMalloc(&dK, hK.size()*4);
        cudaMalloc(&dV, hV.size()*4); cudaMalloc(&dO, hO.size()*4);
        cudaMemcpy(dQ, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dK, hK.data(), hK.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hV.data(), hV.size()*4, cudaMemcpyHostToDevice);
        cuda::kv_cache_attention_decode(dQ, dK, dV, dO, kvlen, nh, nkv, hd);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::kv_cache_attention_decode(dQ, dK, dV, dO, kvlen, nh, nkv, hd);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "kv_cache_attn" << std::setw(18)
                  << ("kv="+std::to_string(kvlen)+" nh=16") << " us=" << std::setw(10)
                  << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    // ============= Paged attention decode =============
    {
        int kvlen=128, nh=16, nkv=8, hd=128, bs=16, max_blocks=64;
        int nb = (kvlen + bs - 1) / bs;
        std::vector<float> hQ(nh*hd), hO(nh*hd);
        std::vector<int> hBT(nb); for(int i=0;i<nb;++i) hBT[i]=i;
        randf(hQ);
        int cache_elems = max_blocks * bs * nkv * hd;
        float *dQ, *dK, *dV, *dO; int *dBT;
        cudaMalloc(&dQ, hQ.size()*4); cudaMalloc(&dK, cache_elems*4);
        cudaMalloc(&dV, cache_elems*4); cudaMalloc(&dO, hO.size()*4);
        cudaMalloc(&dBT, hBT.size()*4);
        cudaMemcpy(dQ, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dBT, hBT.data(), hBT.size()*4, cudaMemcpyHostToDevice);

        int iters = 200;
        cuda::paged_attention_decode(dQ, dK, dV, dBT, dO, kvlen, nh, nkv, hd, bs);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            cuda::paged_attention_decode(dQ, dK, dV, dBT, dO, kvlen, nh, nkv, hd, bs);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "paged_attn_decode" << std::setw(18)
                  << ("kv="+std::to_string(kvlen)+" nh=16") << " us=" << std::setw(10)
                  << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dBT);
    }

    // ============= Transpose =============
    {
        int a=64, b=128, iters=1000;
        int n = a*b;
        std::vector<float> hX(n), hO(n); randf(hX);
        float *dX, *dO;
        cudaMalloc(&dX, n*4); cudaMalloc(&dO, n*4);
        cudaMemcpy(dX, hX.data(), n*4, cudaMemcpyHostToDevice);
        int64_t dims[2]={a,b};
        cuda::transpose(dX, dO, dims, 2, 0, 1); cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) cuda::transpose(dX, dO, dims, 2, 0, 1);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters; volatile float sink = hO[0]; (void)sink;
        std::cout << std::setw(20) << "transpose" << std::setw(18)
                  << "["+std::to_string(a)+","+std::to_string(b)+"]" << " us=" << std::setw(10)
                  << std::fixed << std::setprecision(2) << ms*1000 << "\n";
        cudaFree(dX); cudaFree(dO);
    }

    std::cout << "\nAll CUDA operator benchmarks complete.\n";
}

int main() { run(); return 0; }
