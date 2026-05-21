#pragma once

#include <expected>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/runtime/paged_kv_cache.h"

namespace minillm {

struct PagedAttentionBatch {
    std::vector<int> sequence_ids;
    std::vector<int> sequence_lengths;
    std::vector<int> block_tables;  // [batch_size, max_blocks_per_sequence], padded with -1
    int batch_size = 0;
    int max_blocks_per_sequence = 0;
    int block_size = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
};

class PagedAttentionScheduler {
public:
    explicit PagedAttentionScheduler(PagedKVCache& cache, int max_batch_size);

    Status add_sequence(int sequence_id);
    Status remove_sequence(int sequence_id);
    void clear();

    bool is_active(int sequence_id) const;
    const std::vector<int>& active_sequence_ids() const { return active_sequence_ids_; }
    int max_batch_size() const { return max_batch_size_; }

    std::expected<PagedAttentionBatch, Status> build_batch() const;

    // q:      [batch_size, num_heads, head_dim]
    // output: [batch_size, num_heads, head_dim]
    Status decode_cpu(int layer, const float* q, float* output,
                      int num_heads, int head_dim) const;

private:
    PagedKVCache& cache_;
    int max_batch_size_ = 0;
    std::vector<int> active_sequence_ids_;
};

} // namespace minillm
