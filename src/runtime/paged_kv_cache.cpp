#include "minillm/runtime/paged_kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace minillm {

namespace {

int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

} // namespace

Status PagedKVCache::init(int num_layers, int num_kv_heads, int head_dim,
                          int block_size, int max_blocks) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        block_size <= 0 || max_blocks <= 0) {
        layers_.clear();
        free_blocks_.clear();
        block_used_.clear();
        sequences_.clear();
        num_kv_heads_ = 0;
        head_dim_ = 0;
        block_size_ = 0;
        max_blocks_ = 0;
        return Status::invalid_argument(
            "PagedKVCache init requires positive layers, heads, head_dim, block_size, and max_blocks");
    }

    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;
    max_blocks_ = max_blocks;

    const size_t block_floats = static_cast<size_t>(block_size_) * kv_hidden();
    layers_.resize(static_cast<size_t>(num_layers));
    for (auto& layer : layers_) {
        layer.k.assign(static_cast<size_t>(max_blocks_) * block_floats, 0.0f);
        layer.v.assign(static_cast<size_t>(max_blocks_) * block_floats, 0.0f);
    }

    block_used_.assign(static_cast<size_t>(max_blocks_), false);
    free_blocks_.clear();
    free_blocks_.reserve(static_cast<size_t>(max_blocks_));
    for (int b = max_blocks_ - 1; b >= 0; --b) {
        free_blocks_.push_back(b);
    }
    sequences_.clear();
    return Status::make_ok();
}

void PagedKVCache::reset() {
    for (auto& layer : layers_) {
        std::fill(layer.k.begin(), layer.k.end(), 0.0f);
        std::fill(layer.v.begin(), layer.v.end(), 0.0f);
    }
    block_used_.assign(static_cast<size_t>(max_blocks_), false);
    free_blocks_.clear();
    free_blocks_.reserve(static_cast<size_t>(max_blocks_));
    for (int b = max_blocks_ - 1; b >= 0; --b) {
        free_blocks_.push_back(b);
    }
    sequences_.clear();
}

Status PagedKVCache::ensure_sequence(int sequence_id) {
    if (!initialized()) {
        return Status::runtime_error("PagedKVCache is not initialized");
    }
    if (sequence_id < 0) {
        return Status::invalid_argument("sequence_id must be non-negative");
    }
    if (static_cast<size_t>(sequence_id) >= sequences_.size()) {
        sequences_.resize(static_cast<size_t>(sequence_id) + 1);
    }
    return Status::make_ok();
}

Status PagedKVCache::free_sequence(int sequence_id) {
    if (sequence_id < 0 || static_cast<size_t>(sequence_id) >= sequences_.size()) {
        return Status::make_ok();
    }

    auto& seq = sequences_[static_cast<size_t>(sequence_id)];
    for (int block : seq.blocks) {
        if (block >= 0 && block < max_blocks_ && block_used_[static_cast<size_t>(block)]) {
            block_used_[static_cast<size_t>(block)] = false;
            free_blocks_.push_back(block);
        }
    }
    seq.blocks.clear();
    seq.length = 0;
    return Status::make_ok();
}

bool PagedKVCache::has_sequence(int sequence_id) const {
    return sequence_id >= 0 &&
           static_cast<size_t>(sequence_id) < sequences_.size() &&
           (!sequences_[static_cast<size_t>(sequence_id)].blocks.empty() ||
            sequences_[static_cast<size_t>(sequence_id)].length > 0);
}

int PagedKVCache::sequence_length(int sequence_id) const {
    if (sequence_id < 0 || static_cast<size_t>(sequence_id) >= sequences_.size()) {
        return 0;
    }
    return sequences_[static_cast<size_t>(sequence_id)].length;
}

int PagedKVCache::sequence_capacity(int sequence_id) const {
    if (sequence_id < 0 || static_cast<size_t>(sequence_id) >= sequences_.size()) {
        return 0;
    }
    return static_cast<int>(sequences_[static_cast<size_t>(sequence_id)].blocks.size()) *
           block_size_;
}

const std::vector<int>* PagedKVCache::block_table(int sequence_id) const {
    if (sequence_id < 0 || static_cast<size_t>(sequence_id) >= sequences_.size()) {
        return nullptr;
    }
    return &sequences_[static_cast<size_t>(sequence_id)].blocks;
}

Status PagedKVCache::reserve_sequence(int sequence_id, int total_tokens) {
    auto st = ensure_sequence(sequence_id);
    if (!st.ok()) return st;
    if (total_tokens < 0) {
        return Status::invalid_argument("total_tokens must be non-negative");
    }

    auto& seq = sequences_[static_cast<size_t>(sequence_id)];
    const int needed_blocks = total_tokens == 0 ? 0 : ceil_div_int(total_tokens, block_size_);
    const int current_blocks = static_cast<int>(seq.blocks.size());
    if (needed_blocks <= current_blocks) {
        return Status::make_ok();
    }

    const int to_allocate = needed_blocks - current_blocks;
    if (to_allocate > static_cast<int>(free_blocks_.size())) {
        return Status::out_of_range("PagedKVCache does not have enough free blocks");
    }

    for (int i = 0; i < to_allocate; ++i) {
        int block = free_blocks_.back();
        free_blocks_.pop_back();
        block_used_[static_cast<size_t>(block)] = true;
        seq.blocks.push_back(block);
    }
    return Status::make_ok();
}

Status PagedKVCache::set_sequence_length(int sequence_id, int length) {
    if (length < 0) {
        return Status::invalid_argument("sequence length must be non-negative");
    }
    auto st = reserve_sequence(sequence_id, length);
    if (!st.ok()) return st;
    sequences_[static_cast<size_t>(sequence_id)].length = length;
    return Status::make_ok();
}

std::expected<int, Status> PagedKVCache::physical_block_for_position(
    int sequence_id, int token_pos) const {
    if (sequence_id < 0 || static_cast<size_t>(sequence_id) >= sequences_.size()) {
        return std::unexpected(Status::not_found("sequence not found in PagedKVCache"));
    }
    if (token_pos < 0) {
        return std::unexpected(Status::out_of_range("token position must be non-negative"));
    }
    const auto& seq = sequences_[static_cast<size_t>(sequence_id)];
    const int logical_block = token_pos / block_size_;
    if (logical_block < 0 || static_cast<size_t>(logical_block) >= seq.blocks.size()) {
        return std::unexpected(Status::out_of_range(
            "token position is outside the sequence block table"));
    }
    return seq.blocks[static_cast<size_t>(logical_block)];
}

std::expected<float*, Status> PagedKVCache::mutable_token_ptr(
    bool key, int sequence_id, int layer, int token_pos) {
    if (layer < 0 || layer >= num_layers()) {
        return std::unexpected(Status::out_of_range("PagedKVCache layer index out of range"));
    }
    auto block = physical_block_for_position(sequence_id, token_pos);
    if (!block) return std::unexpected(block.error());

    const int offset = token_pos % block_size_;
    const size_t block_stride = static_cast<size_t>(block_size_) * kv_hidden();
    const size_t index = static_cast<size_t>(*block) * block_stride +
                         static_cast<size_t>(offset) * kv_hidden();
    auto& storage = key ? layers_[static_cast<size_t>(layer)].k
                        : layers_[static_cast<size_t>(layer)].v;
    return storage.data() + index;
}

std::expected<const float*, Status> PagedKVCache::token_ptr(
    bool key, int sequence_id, int layer, int token_pos, int kv_head) const {
    if (layer < 0 || layer >= num_layers()) {
        return std::unexpected(Status::out_of_range("PagedKVCache layer index out of range"));
    }
    if (kv_head < 0 || kv_head >= num_kv_heads_) {
        return std::unexpected(Status::out_of_range("PagedKVCache KV head index out of range"));
    }
    if (token_pos < 0 || token_pos >= sequence_length(sequence_id)) {
        return std::unexpected(Status::out_of_range(
            "token position is outside the sequence length"));
    }

    auto block = physical_block_for_position(sequence_id, token_pos);
    if (!block) return std::unexpected(block.error());

    const int offset = token_pos % block_size_;
    const size_t block_stride = static_cast<size_t>(block_size_) * kv_hidden();
    const size_t index = static_cast<size_t>(*block) * block_stride +
                         static_cast<size_t>(offset) * kv_hidden() +
                         static_cast<size_t>(kv_head) * head_dim_;
    const auto& storage = key ? layers_[static_cast<size_t>(layer)].k
                              : layers_[static_cast<size_t>(layer)].v;
    return storage.data() + index;
}

Status PagedKVCache::write_tokens(int sequence_id, int layer, int start_pos,
                                  const float* k, const float* v, int token_count) {
    if (!k || !v) {
        return Status::invalid_argument("PagedKVCache write_tokens requires non-null K/V");
    }
    if (start_pos < 0 || token_count < 0) {
        return Status::invalid_argument("start_pos and token_count must be non-negative");
    }
    if (layer < 0 || layer >= num_layers()) {
        return Status::out_of_range("PagedKVCache layer index out of range");
    }
    if (token_count == 0) {
        return Status::make_ok();
    }

    const int end_pos = start_pos + token_count;
    auto st = reserve_sequence(sequence_id, end_pos);
    if (!st.ok()) return st;

    const int row_floats = kv_hidden();
    for (int t = 0; t < token_count; ++t) {
        auto dst_k = mutable_token_ptr(true, sequence_id, layer, start_pos + t);
        if (!dst_k) return dst_k.error();
        auto dst_v = mutable_token_ptr(false, sequence_id, layer, start_pos + t);
        if (!dst_v) return dst_v.error();
        std::memcpy(*dst_k, k + static_cast<size_t>(t) * row_floats,
                    static_cast<size_t>(row_floats) * sizeof(float));
        std::memcpy(*dst_v, v + static_cast<size_t>(t) * row_floats,
                    static_cast<size_t>(row_floats) * sizeof(float));
    }

    auto& seq = sequences_[static_cast<size_t>(sequence_id)];
    seq.length = std::max(seq.length, end_pos);
    return Status::make_ok();
}

Status PagedKVCache::append_tokens(int sequence_id, int layer,
                                   const float* k, const float* v, int token_count) {
    const int start_pos = sequence_length(sequence_id);
    return write_tokens(sequence_id, layer, start_pos, k, v, token_count);
}

std::expected<const float*, Status> PagedKVCache::key_ptr(
    int sequence_id, int layer, int token_pos, int kv_head) const {
    return token_ptr(true, sequence_id, layer, token_pos, kv_head);
}

std::expected<const float*, Status> PagedKVCache::value_ptr(
    int sequence_id, int layer, int token_pos, int kv_head) const {
    return token_ptr(false, sequence_id, layer, token_pos, kv_head);
}

Status paged_attention_decode(const PagedKVCache& cache, int sequence_id,
                              int layer, const float* q, float* output,
                              int num_heads, int head_dim) {
    if (!cache.initialized()) {
        return Status::runtime_error("PagedKVCache is not initialized");
    }
    if (!q || !output) {
        return Status::invalid_argument("paged_attention_decode requires non-null q/output");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        return Status::invalid_argument("num_heads and head_dim must be positive");
    }
    if (head_dim != cache.head_dim()) {
        return Status::shape_mismatch("paged attention head_dim must match cache head_dim");
    }
    if (cache.num_kv_heads() <= 0 || num_heads % cache.num_kv_heads() != 0) {
        return Status::invalid_argument("num_heads must be divisible by cache num_kv_heads");
    }

    const int kv_len = cache.sequence_length(sequence_id);
    if (kv_len <= 0) {
        return Status::invalid_argument("paged_attention_decode requires a non-empty sequence");
    }

    const int group_size = num_heads / cache.num_kv_heads();
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(kv_len));

    for (int h = 0; h < num_heads; ++h) {
        const int kv_h = h / group_size;
        const float* q_vec = q + static_cast<size_t>(h) * head_dim;

        float max_score = -std::numeric_limits<float>::max();
        for (int pos = 0; pos < kv_len; ++pos) {
            auto k_vec = cache.key_ptr(sequence_id, layer, pos, kv_h);
            if (!k_vec) return k_vec.error();

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * (*k_vec)[d];
            }
            float score = dot * scale;
            scores[static_cast<size_t>(pos)] = score;
            max_score = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (int pos = 0; pos < kv_len; ++pos) {
            float p = std::exp(scores[static_cast<size_t>(pos)] - max_score);
            scores[static_cast<size_t>(pos)] = p;
            sum += p;
        }
        if (sum <= 0.0f) {
            return Status::runtime_error("paged attention softmax sum is non-positive");
        }

        float* out_vec = output + static_cast<size_t>(h) * head_dim;
        std::fill(out_vec, out_vec + head_dim, 0.0f);
        for (int pos = 0; pos < kv_len; ++pos) {
            auto v_vec = cache.value_ptr(sequence_id, layer, pos, kv_h);
            if (!v_vec) return v_vec.error();
            const float weight = scores[static_cast<size_t>(pos)] / sum;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += weight * (*v_vec)[d];
            }
        }
    }

    return Status::make_ok();
}

} // namespace minillm
