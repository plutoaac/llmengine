#pragma once

#include <vector>

#include "minillm/core/device.h"
#include "minillm/core/status.h"

namespace minillm {

struct KVCacheLayer {
    std::vector<float> k;  // [max_seq_len * kv_hidden], row-major
    std::vector<float> v;  // [max_seq_len * kv_hidden], row-major
    float* cuda_k = nullptr;
    float* cuda_v = nullptr;
    size_t cuda_bytes = 0;
};

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    void init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len);
    Status init_cuda(int num_layers, int num_kv_heads, int head_dim,
                     int max_seq_len, int device_index = 0);

    float* k_data(int layer);
    float* v_data(int layer);
    int num_layers() const { return static_cast<int>(layers_.size()); }
    int kv_hidden() const { return num_kv_heads_ * head_dim_; }
    int cached_len() const { return cached_len_; }
    void set_cached_len(int len);
    void advance(int n = 1) { set_cached_len(cached_len_ + n); }
    bool can_append(int n) const;
    void reset();
    Status release();
    int max_seq_len() const { return max_seq_len_; }
    bool initialized() const { return max_seq_len_ > 0; }
    const Device& device() const { return device_; }
    bool is_cuda() const { return device_.type == DeviceType::CUDA; }

private:
    std::vector<KVCacheLayer> layers_;
    Device device_ = Device::cpu();
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_len_ = 0;
    int cached_len_ = 0;
};

} // namespace minillm
