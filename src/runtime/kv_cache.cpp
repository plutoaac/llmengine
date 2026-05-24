#include "minillm/runtime/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <string>

#if defined(MINILLM_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace minillm {

KVCache::~KVCache() {
    (void)release();
}

void KVCache::init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len) {
    (void)release();
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) {
        layers_.clear();
        num_kv_heads_ = 0;
        head_dim_ = 0;
        max_seq_len_ = 0;
        cached_len_ = 0;
        return;
    }

    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    cached_len_ = 0;
    device_ = Device::cpu();

    int kv_h = num_kv_heads * head_dim;
    layers_.resize(num_layers);
    for (auto& layer : layers_) {
        layer.k.resize(static_cast<size_t>(max_seq_len) * kv_h, 0.0f);
        layer.v.resize(static_cast<size_t>(max_seq_len) * kv_h, 0.0f);
    }
}

Status KVCache::init_cuda(int num_layers, int num_kv_heads, int head_dim,
                          int max_seq_len, int device_index) {
    auto st = release();
    if (!st.ok()) return st;
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) {
        return Status::invalid_argument(
            "KVCache CUDA init requires positive layers, heads, head_dim, and max_seq_len");
    }

#if defined(MINILLM_ENABLE_CUDA)
    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        return Status::runtime_error(
            "cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
    }

    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    cached_len_ = 0;
    device_ = Device::cuda(device_index);

    const size_t bytes =
        static_cast<size_t>(max_seq_len_) * kv_hidden() * sizeof(float);
    layers_.resize(static_cast<size_t>(num_layers));
    for (auto& layer : layers_) {
        err = cudaMalloc(&layer.cuda_k, bytes);
        if (err != cudaSuccess) {
            (void)release();
            return Status::runtime_error(
                "cudaMalloc KV cache K failed: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&layer.cuda_v, bytes);
        if (err != cudaSuccess) {
            (void)release();
            return Status::runtime_error(
                "cudaMalloc KV cache V failed: " + std::string(cudaGetErrorString(err)));
        }
        layer.cuda_bytes = bytes;
        err = cudaMemset(layer.cuda_k, 0, bytes);
        if (err != cudaSuccess) {
            (void)release();
            return Status::runtime_error(
                "cudaMemset KV cache K failed: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaMemset(layer.cuda_v, 0, bytes);
        if (err != cudaSuccess) {
            (void)release();
            return Status::runtime_error(
                "cudaMemset KV cache V failed: " + std::string(cudaGetErrorString(err)));
        }
    }
    return Status::make_ok();
#else
    return Status::unsupported("MiniLLMEngine was built without CUDA support");
#endif
}

float* KVCache::k_data(int layer) {
    if (is_cuda()) return layers_[layer].cuda_k;
    return layers_[layer].k.data();
}

float* KVCache::v_data(int layer) {
    if (is_cuda()) return layers_[layer].cuda_v;
    return layers_[layer].v.data();
}

void KVCache::set_cached_len(int len) {
    cached_len_ = std::clamp(len, 0, max_seq_len_);
}

bool KVCache::can_append(int n) const {
    return n >= 0 && cached_len_ <= max_seq_len_ && n <= max_seq_len_ - cached_len_;
}

void KVCache::reset() {
    cached_len_ = 0;
    for (auto& layer : layers_) {
        if (is_cuda()) {
#if defined(MINILLM_ENABLE_CUDA)
            if (layer.cuda_k && layer.cuda_bytes > 0) {
                cudaMemset(layer.cuda_k, 0, layer.cuda_bytes);
            }
            if (layer.cuda_v && layer.cuda_bytes > 0) {
                cudaMemset(layer.cuda_v, 0, layer.cuda_bytes);
            }
#endif
        } else {
            if (!layer.k.empty()) {
                std::memset(layer.k.data(), 0, layer.k.size() * sizeof(float));
            }
            if (!layer.v.empty()) {
                std::memset(layer.v.data(), 0, layer.v.size() * sizeof(float));
            }
        }
    }
}

Status KVCache::release() {
#if defined(MINILLM_ENABLE_CUDA)
    for (auto& layer : layers_) {
        if (layer.cuda_k) {
            cudaError_t err = cudaFree(layer.cuda_k);
            layer.cuda_k = nullptr;
            if (err != cudaSuccess) {
                return Status::runtime_error(
                    "cudaFree KV cache K failed: " + std::string(cudaGetErrorString(err)));
            }
        }
        if (layer.cuda_v) {
            cudaError_t err = cudaFree(layer.cuda_v);
            layer.cuda_v = nullptr;
            if (err != cudaSuccess) {
                return Status::runtime_error(
                    "cudaFree KV cache V failed: " + std::string(cudaGetErrorString(err)));
            }
        }
        layer.cuda_bytes = 0;
    }
#endif
    layers_.clear();
    num_kv_heads_ = 0;
    head_dim_ = 0;
    max_seq_len_ = 0;
    cached_len_ = 0;
    device_ = Device::cpu();
    return Status::make_ok();
}

} // namespace minillm
