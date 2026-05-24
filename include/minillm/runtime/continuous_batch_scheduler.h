#pragma once

#include <optional>
#include <string>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/runtime/paged_kv_cache.h"

namespace minillm {

struct SequenceRequest {
    int sequence_id = -1;
    std::vector<int32_t> prompt_tokens;
    int max_tokens = 64;
};

struct SequenceOutput {
    int sequence_id = -1;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    bool finished = false;
};

enum class SeqPhase {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
};

struct SeqState {
    int sequence_id = -1;
    SeqPhase phase = SeqPhase::Waiting;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int max_tokens = 64;
    int prefilled_tokens = 0;
};

class ContinuousBatchScheduler {
public:
    struct Config {
        int max_batch_size = 4;
    };

    ContinuousBatchScheduler(PagedKVCache& cache, Config config);

    Status submit(const SequenceRequest& request);

    Status admit_waiting();

    const std::vector<int>& active_ids() const { return active_ids_; }
    const std::vector<int>& waiting_ids() const { return waiting_queue_; }

    SeqState& state(int sequence_id);
    const SeqState& state(int sequence_id) const;

    void mark_prefill_progress(int sequence_id, int tokens_prefilled);
    void mark_token_generated(int sequence_id, int32_t token);
    void mark_finished(int sequence_id);

    void evict_finished();

    bool has_active() const { return !active_ids_.empty(); }
    bool has_waiting() const { return !waiting_queue_.empty(); }
    bool all_done() const { return active_ids_.empty() && waiting_queue_.empty(); }

    std::vector<SequenceOutput> collect_finished();

    int active_count() const { return static_cast<int>(active_ids_.size()); }
    int waiting_count() const { return static_cast<int>(waiting_queue_.size()); }

    PagedKVCache& cache() { return cache_; }

private:
    PagedKVCache& cache_;
    Config config_;
    std::vector<SeqState> states_;
    std::vector<int> waiting_queue_;
    std::vector<int> active_ids_;
    int next_sequence_id_ = 0;
};

} // namespace minillm
