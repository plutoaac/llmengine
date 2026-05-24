#pragma once

#include <expected>
#include <vector>

#include "minillm/core/status.h"

namespace minillm {

struct PagedKVCacheLayer {
    std::vector<float> k;  // [max_blocks, block_size, num_kv_heads * head_dim]
    std::vector<float> v;
#if defined(MINILLM_ENABLE_CUDA)
    float* cuda_k = nullptr;
    float* cuda_v = nullptr;
#endif
};

struct PagedKVSequence {
    int length = 0;
    std::vector<int> blocks;
};

class PagedKVCache {
public:
    PagedKVCache() = default;
    ~PagedKVCache();

    Status init(int num_layers, int num_kv_heads, int head_dim,
                int block_size, int max_blocks);
#if defined(MINILLM_ENABLE_CUDA)
    Status init_cuda(int num_layers, int num_kv_heads, int head_dim,
                     int block_size, int max_blocks, int device_index = 0);
    float* cuda_k_data(int layer);
    float* cuda_v_data(int layer);
    const int* device_block_table(int sequence_id) const;
    bool is_cuda() const;
#endif
    void reset();
    Status release();

    bool initialized() const { return block_size_ > 0 && max_blocks_ > 0; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    int kv_hidden() const { return num_kv_heads_ * head_dim_; }
    int block_size() const { return block_size_; }
    int max_blocks() const { return max_blocks_; }
    int free_block_count() const { return static_cast<int>(free_blocks_.size()); }

    Status ensure_sequence(int sequence_id);
    Status free_sequence(int sequence_id);
    bool has_sequence(int sequence_id) const;
    int sequence_length(int sequence_id) const;
    int sequence_capacity(int sequence_id) const;
    const std::vector<int>* block_table(int sequence_id) const;

    Status reserve_sequence(int sequence_id, int total_tokens);
    Status set_sequence_length(int sequence_id, int length);

    Status write_tokens(int sequence_id, int layer, int start_pos,
                        const float* k, const float* v, int token_count);
    Status append_tokens(int sequence_id, int layer,
                         const float* k, const float* v, int token_count);
#if defined(MINILLM_ENABLE_CUDA)
    Status write_tokens_cuda(int sequence_id, int layer, int start_pos,
                             const float* k, const float* v, int token_count);
    Status upload_block_table(int sequence_id);
#endif

    std::expected<const float*, Status> key_ptr(
        int sequence_id, int layer, int token_pos, int kv_head) const;
    std::expected<const float*, Status> value_ptr(
        int sequence_id, int layer, int token_pos, int kv_head) const;

private:
    std::expected<int, Status> physical_block_for_position(
        int sequence_id, int token_pos) const;
    std::expected<float*, Status> mutable_token_ptr(
        bool key, int sequence_id, int layer, int token_pos);
    std::expected<const float*, Status> token_ptr(
        bool key, int sequence_id, int layer, int token_pos, int kv_head) const;

    Status write_tokens_impl(int sequence_id, int layer, int start_pos,
                             const float* k, const float* v, int token_count, bool use_device);

    std::vector<PagedKVCacheLayer> layers_;
    std::vector<int> free_blocks_;
    std::vector<bool> block_used_;
    std::vector<PagedKVSequence> sequences_;

    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int block_size_ = 0;
    int max_blocks_ = 0;
#if defined(MINILLM_ENABLE_CUDA)
    std::vector<int*> cuda_block_tables_;
    std::vector<size_t> cuda_block_table_sizes_;
    int cuda_device_ = 0;
#endif
};

Status paged_attention_decode(const PagedKVCache& cache, int sequence_id,
                              int layer, const float* q, float* output,
                              int num_heads, int head_dim);

} // namespace minillm
