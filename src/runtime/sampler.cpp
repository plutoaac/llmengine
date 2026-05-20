#include "minillm/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minillm {

Sampler::Sampler(uint32_t seed) : rng_(seed) {}

int32_t Sampler::sample(const float* logits, int32_t vocab_size,
                         const SamplingConfig& cfg,
                         const std::vector<int32_t>& input_ids) {
    if (!logits || vocab_size <= 0) {
        return -1;
    }

    // Make a mutable copy for modifications
    std::vector<float> probs(logits, logits + vocab_size);

    // Apply repetition penalty
    if (cfg.repetition_penalty != 1.0f && !input_ids.empty()) {
        for (int32_t id : input_ids) {
            if (id >= 0 && id < vocab_size) {
                if (probs[id] > 0.0f) {
                    probs[id] /= cfg.repetition_penalty;
                } else {
                    probs[id] *= cfg.repetition_penalty;
                }
            }
        }
    }

    // Apply temperature
    if (cfg.temperature != 1.0f && cfg.temperature > 0.0f) {
        float inv_temp = 1.0f / cfg.temperature;
        for (float& p : probs) p *= inv_temp;
    }

    // Find max for numerical stability
    float max_val = *std::max_element(probs.begin(), probs.end());

    // Compute exp(logit - max) and sum
    std::vector<float> exp_vals(vocab_size);
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        exp_vals[i] = std::exp(probs[i] - max_val);
        sum += exp_vals[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (float& v : exp_vals) v *= inv_sum;

    // Build indexed array for sorting
    struct TokenProb { int32_t id; float prob; };
    std::vector<TokenProb> candidates;
    candidates.reserve(vocab_size);

    // Top-k filtering
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        for (int32_t i = 0; i < vocab_size; ++i) {
            candidates.push_back({i, exp_vals[i]});
        }
        // Partial sort: keep top-k
        std::partial_sort(candidates.begin(), candidates.begin() + cfg.top_k,
                          candidates.end(),
                          [](const TokenProb& a, const TokenProb& b) {
                              return a.prob > b.prob;
                          });
        candidates.resize(cfg.top_k);
    } else {
        for (int32_t i = 0; i < vocab_size; ++i) {
            candidates.push_back({i, exp_vals[i]});
        }
        // Sort by probability descending for top-p
        std::sort(candidates.begin(), candidates.end(),
                  [](const TokenProb& a, const TokenProb& b) {
                      return a.prob > b.prob;
                  });
    }

    // Top-p (nucleus) filtering
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        float cumsum = 0.0f;
        int32_t cutoff = static_cast<int32_t>(candidates.size());
        for (int32_t i = 0; i < static_cast<int32_t>(candidates.size()); ++i) {
            cumsum += candidates[i].prob;
            if (cumsum >= cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
    }

    // Greedy: return argmax
    if (cfg.greedy) {
        int32_t best = candidates[0].id;
        float best_prob = candidates[0].prob;
        for (const auto& c : candidates) {
            if (c.prob > best_prob) {
                best_prob = c.prob;
                best = c.id;
            }
        }
        return best;
    }

    // Stochastic: renormalize and sample
    float renorm_sum = 0.0f;
    for (const auto& c : candidates) renorm_sum += c.prob;
    float inv_renorm = 1.0f / renorm_sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    for (const auto& c : candidates) {
        cumsum += c.prob * inv_renorm;
        if (cumsum >= r) return c.id;
    }
    return candidates.back().id;
}

std::vector<std::pair<int32_t, float>> Sampler::get_top_tokens(
    const float* logits, int32_t vocab_size,
    const SamplingConfig& cfg, int32_t top_n) {
    if (!logits || vocab_size <= 0 || top_n <= 0) {
        return {};
    }

    // Apply temperature
    std::vector<float> scaled(logits, logits + vocab_size);
    if (cfg.temperature > 0.0f && cfg.temperature != 1.0f) {
        float inv_temp = 1.0f / cfg.temperature;
        for (float& v : scaled) v *= inv_temp;
    }

    // Softmax
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> exp_vals(vocab_size);
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        exp_vals[i] = std::exp(scaled[i] - max_val);
        sum += exp_vals[i];
    }
    float inv_sum = 1.0f / sum;

    struct TokenProb { int32_t id; float prob; };
    std::vector<TokenProb> candidates;
    candidates.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        candidates.push_back({i, exp_vals[i] * inv_sum});
    }

    int32_t n = std::min(top_n, vocab_size);
    std::partial_sort(candidates.begin(), candidates.begin() + n,
                      candidates.end(),
                      [](const TokenProb& a, const TokenProb& b) {
                          return a.prob > b.prob;
                      });

    std::vector<std::pair<int32_t, float>> result;
    for (int32_t i = 0; i < n; ++i) {
        result.push_back({candidates[i].id, candidates[i].prob});
    }
    return result;
}

} // namespace minillm
