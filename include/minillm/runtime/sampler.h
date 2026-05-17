#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace minillm {

struct SamplingConfig {
    // Greedy: just argmax
    bool greedy = true;
    // Top-k: only sample from top k tokens (0 = disabled)
    int32_t top_k = 0;
    // Top-p (nucleus): sample from smallest set of tokens with cumulative prob >= top_p (0 = disabled)
    float top_p = 0.0f;
    // Temperature: scale logits before softmax (1.0 = no scaling)
    float temperature = 1.0f;
    // Repetition penalty: penalize tokens that appeared in the input
    float repetition_penalty = 1.0f;
};

class Sampler {
public:
    explicit Sampler(uint32_t seed = 42);

    // Sample a token from logits.
    // input_ids: previously generated token IDs (for repetition penalty)
    int32_t sample(const float* logits, int32_t vocab_size,
                   const SamplingConfig& cfg,
                   const std::vector<int32_t>& input_ids = {});

    // Get the probability distribution after applying temperature/top-k/top-p
    std::vector<std::pair<int32_t, float>> get_top_tokens(
        const float* logits, int32_t vocab_size,
        const SamplingConfig& cfg, int32_t top_n);

private:
    std::mt19937 rng_;
};

} // namespace minillm
