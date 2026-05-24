#include "minillm/runtime/continuous_batch_scheduler.h"

#include <algorithm>
#include <string>

namespace minillm {

ContinuousBatchScheduler::ContinuousBatchScheduler(PagedKVCache& cache, Config config)
    : cache_(cache), config_(std::move(config)) {}

SeqState& ContinuousBatchScheduler::state(int sequence_id) {
    return states_.at(static_cast<size_t>(sequence_id));
}

const SeqState& ContinuousBatchScheduler::state(int sequence_id) const {
    return states_.at(static_cast<size_t>(sequence_id));
}

Status ContinuousBatchScheduler::submit(const SequenceRequest& request) {
    if (request.prompt_tokens.empty()) {
        return Status::invalid_argument("SequenceRequest must have at least one prompt token");
    }
    if (request.max_tokens <= 0) {
        return Status::invalid_argument("SequenceRequest max_tokens must be positive");
    }

    const int seq_id = request.sequence_id >= 0 ? request.sequence_id : next_sequence_id_++;
    if (seq_id < 0) {
        return Status::invalid_argument("SequenceRequest sequence_id must be non-negative");
    }
    if (request.sequence_id >= 0 && seq_id >= next_sequence_id_) {
        next_sequence_id_ = seq_id + 1;
    }
    if (static_cast<size_t>(seq_id) >= states_.size()) {
        states_.resize(static_cast<size_t>(seq_id) + 1);
    }

    auto& s = states_[static_cast<size_t>(seq_id)];
    if (s.sequence_id >= 0) {
        return Status::invalid_argument(
            "SequenceRequest sequence_id is already in use: " + std::to_string(seq_id));
    }

    s.sequence_id = seq_id;
    s.phase = SeqPhase::Waiting;
    s.prompt_tokens = request.prompt_tokens;
    s.generated_tokens.clear();
    s.max_tokens = request.max_tokens;
    s.prefilled_tokens = 0;

    waiting_queue_.push_back(seq_id);
    return Status::make_ok();
}

Status ContinuousBatchScheduler::admit_waiting() {
    if (config_.max_batch_size <= 0) {
        return Status::invalid_argument("ContinuousBatchScheduler max_batch_size must be positive");
    }
    while (!waiting_queue_.empty()) {
        if (static_cast<int>(active_ids_.size()) >= config_.max_batch_size) break;

        int seq_id = waiting_queue_.front();
        auto& s = state(seq_id);

        auto st = cache_.ensure_sequence(seq_id);
        if (!st.ok()) return st;

        const int total_tokens = static_cast<int>(s.prompt_tokens.size()) + s.max_tokens;
        st = cache_.reserve_sequence(seq_id, total_tokens);
        if (!st.ok()) break;

        s.phase = SeqPhase::Prefilling;
        active_ids_.push_back(seq_id);
        waiting_queue_.erase(waiting_queue_.begin());
    }
    return Status::make_ok();
}

void ContinuousBatchScheduler::mark_prefill_progress(int sequence_id, int tokens_prefilled) {
    auto& s = state(sequence_id);
    s.prefilled_tokens = tokens_prefilled;
    if (s.prefilled_tokens >= static_cast<int>(s.prompt_tokens.size())) {
        s.phase = SeqPhase::Decoding;
    }
}

void ContinuousBatchScheduler::mark_token_generated(int sequence_id, int32_t token) {
    auto& s = state(sequence_id);
    s.generated_tokens.push_back(token);
    if (static_cast<int>(s.generated_tokens.size()) >= s.max_tokens) {
        s.phase = SeqPhase::Finished;
    }
}

void ContinuousBatchScheduler::mark_finished(int sequence_id) {
    state(sequence_id).phase = SeqPhase::Finished;
}

void ContinuousBatchScheduler::evict_finished() {
    auto it = active_ids_.begin();
    while (it != active_ids_.end()) {
        if (state(*it).phase == SeqPhase::Finished) {
            cache_.free_sequence(*it);
            it = active_ids_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<SequenceOutput> ContinuousBatchScheduler::collect_finished() {
    std::vector<SequenceOutput> result;
    for (auto& s : states_) {
        if (s.phase == SeqPhase::Finished && s.sequence_id >= 0) {
            SequenceOutput out;
            out.sequence_id = s.sequence_id;
            out.prompt_tokens = s.prompt_tokens;
            out.generated_tokens = s.generated_tokens;
            out.finished = true;
            result.push_back(std::move(out));
            s.sequence_id = -1;
        }
    }
    return result;
}

} // namespace minillm
