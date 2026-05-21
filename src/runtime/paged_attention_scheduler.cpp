#include "minillm/runtime/paged_attention_scheduler.h"

#include <algorithm>
#include <string>

namespace minillm {

namespace {

int blocks_for_length(int length, int block_size) {
    return (length + block_size - 1) / block_size;
}

} // namespace

PagedAttentionScheduler::PagedAttentionScheduler(PagedKVCache& cache, int max_batch_size)
    : cache_(cache), max_batch_size_(max_batch_size) {}

Status PagedAttentionScheduler::add_sequence(int sequence_id) {
    if (max_batch_size_ <= 0) {
        return Status::invalid_argument("PagedAttentionScheduler max_batch_size must be positive");
    }
    if (sequence_id < 0) {
        return Status::invalid_argument("sequence_id must be non-negative");
    }
    if (!cache_.has_sequence(sequence_id) || cache_.sequence_length(sequence_id) <= 0) {
        return Status::not_found(
            "sequence " + std::to_string(sequence_id) + " is not present in PagedKVCache");
    }
    if (is_active(sequence_id)) {
        return Status::make_ok();
    }
    if (static_cast<int>(active_sequence_ids_.size()) >= max_batch_size_) {
        return Status::out_of_range("PagedAttentionScheduler active batch is full");
    }
    active_sequence_ids_.push_back(sequence_id);
    return Status::make_ok();
}

Status PagedAttentionScheduler::remove_sequence(int sequence_id) {
    auto it = std::find(active_sequence_ids_.begin(), active_sequence_ids_.end(), sequence_id);
    if (it != active_sequence_ids_.end()) {
        active_sequence_ids_.erase(it);
    }
    return Status::make_ok();
}

void PagedAttentionScheduler::clear() {
    active_sequence_ids_.clear();
}

bool PagedAttentionScheduler::is_active(int sequence_id) const {
    return std::find(active_sequence_ids_.begin(), active_sequence_ids_.end(), sequence_id) !=
           active_sequence_ids_.end();
}

std::expected<PagedAttentionBatch, Status> PagedAttentionScheduler::build_batch() const {
    if (active_sequence_ids_.empty()) {
        return std::unexpected(Status::invalid_argument(
            "PagedAttentionScheduler cannot build an empty batch"));
    }
    if (static_cast<int>(active_sequence_ids_.size()) > max_batch_size_) {
        return std::unexpected(Status::internal_error(
            "PagedAttentionScheduler active batch exceeds max_batch_size"));
    }

    PagedAttentionBatch batch;
    batch.batch_size = static_cast<int>(active_sequence_ids_.size());
    batch.block_size = cache_.block_size();
    batch.num_kv_heads = cache_.num_kv_heads();
    batch.head_dim = cache_.head_dim();
    batch.sequence_ids = active_sequence_ids_;
    batch.sequence_lengths.reserve(active_sequence_ids_.size());

    for (int sequence_id : active_sequence_ids_) {
        const int length = cache_.sequence_length(sequence_id);
        const auto* table = cache_.block_table(sequence_id);
        if (!table || length <= 0) {
            return std::unexpected(Status::not_found(
                "active sequence " + std::to_string(sequence_id) +
                " is missing from PagedKVCache"));
        }
        batch.sequence_lengths.push_back(length);
        const int needed_blocks = blocks_for_length(length, batch.block_size);
        if (static_cast<int>(table->size()) < needed_blocks) {
            return std::unexpected(Status::internal_error(
                "active sequence " + std::to_string(sequence_id) +
                " has a shorter block table than its length requires"));
        }
        batch.max_blocks_per_sequence = std::max(
            batch.max_blocks_per_sequence, needed_blocks);
    }

    if (batch.max_blocks_per_sequence <= 0) {
        return std::unexpected(Status::runtime_error(
            "PagedAttentionScheduler active sequences have no blocks"));
    }

    batch.block_tables.assign(
        static_cast<size_t>(batch.batch_size) * batch.max_blocks_per_sequence, -1);
    for (int b = 0; b < batch.batch_size; ++b) {
        const int sequence_id = batch.sequence_ids[static_cast<size_t>(b)];
        const auto* table = cache_.block_table(sequence_id);
        const int needed_blocks =
            blocks_for_length(batch.sequence_lengths[static_cast<size_t>(b)],
                              batch.block_size);
        for (int i = 0; i < needed_blocks; ++i) {
            batch.block_tables[static_cast<size_t>(b) * batch.max_blocks_per_sequence + i] =
                (*table)[static_cast<size_t>(i)];
        }
    }

    return batch;
}

Status PagedAttentionScheduler::decode_cpu(int layer, const float* q, float* output,
                                           int num_heads, int head_dim) const {
    if (!q || !output) {
        return Status::invalid_argument("decode_cpu requires non-null q/output");
    }
    auto batch = build_batch();
    if (!batch) return batch.error();
    if (head_dim != batch->head_dim) {
        return Status::shape_mismatch("decode_cpu head_dim must match PagedKVCache head_dim");
    }

    const int stride = num_heads * head_dim;
    for (int b = 0; b < batch->batch_size; ++b) {
        const int sequence_id = batch->sequence_ids[static_cast<size_t>(b)];
        auto st = paged_attention_decode(cache_, sequence_id, layer,
                                          q + static_cast<size_t>(b) * stride,
                                          output + static_cast<size_t>(b) * stride,
                                          num_heads, head_dim);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

} // namespace minillm
