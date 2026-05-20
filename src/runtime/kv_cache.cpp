#include "minillm/runtime/kv_cache.h"

#include <algorithm>
#include <cstring>

namespace minillm {

void KVCache::init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len) {
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

    int kv_h = num_kv_heads * head_dim;
    layers_.resize(num_layers);
    for (auto& layer : layers_) {
        layer.k.resize(static_cast<size_t>(max_seq_len) * kv_h, 0.0f);
        layer.v.resize(static_cast<size_t>(max_seq_len) * kv_h, 0.0f);
    }
}

float* KVCache::k_data(int layer) {
    return layers_[layer].k.data();
}

float* KVCache::v_data(int layer) {
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
        if (!layer.k.empty()) {
            std::memset(layer.k.data(), 0, layer.k.size() * sizeof(float));
        }
        if (!layer.v.empty()) {
            std::memset(layer.v.data(), 0, layer.v.size() * sizeof(float));
        }
    }
}

} // namespace minillm
