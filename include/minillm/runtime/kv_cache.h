#pragma once

#include <vector>

namespace minillm {

struct KVCacheLayer {
    std::vector<float> k;  // [max_seq_len * kv_hidden], row-major
    std::vector<float> v;  // [max_seq_len * kv_hidden], row-major
};

class KVCache {
public:
    KVCache() = default;

    void init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len);

    float* k_data(int layer);
    float* v_data(int layer);
    int num_layers() const { return static_cast<int>(layers_.size()); }
    int kv_hidden() const { return num_kv_heads_ * head_dim_; }
    int cached_len() const { return cached_len_; }
    void set_cached_len(int len);
    void advance(int n = 1) { set_cached_len(cached_len_ + n); }
    bool can_append(int n) const;
    void reset();
    int max_seq_len() const { return max_seq_len_; }
    bool initialized() const { return max_seq_len_ > 0; }

private:
    std::vector<KVCacheLayer> layers_;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_len_ = 0;
    int cached_len_ = 0;
};

} // namespace minillm
