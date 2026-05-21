#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "minillm/minillm.h"

using namespace minillm;

static void assert_near(float actual, float expected, float tol = 1e-5f) {
    assert(std::abs(actual - expected) <= tol);
}

static std::vector<float> reference_softmax(const std::vector<float>& x) {
    float max_val = -std::numeric_limits<float>::max();
    for (float v : x) max_val = std::max(max_val, v);

    std::vector<float> y(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - max_val);
        sum += y[i];
    }
    for (float& v : y) v /= sum;
    return y;
}

static std::vector<float> reference_decode_head(
    const float* q, const std::vector<float>& k, const std::vector<float>& v,
    int kv_len, int num_kv_heads, int head_dim, int kv_head) {
    const int kv_hidden = num_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(kv_len));

    for (int pos = 0; pos < kv_len; ++pos) {
        const float* k_vec =
            k.data() + static_cast<size_t>(pos) * kv_hidden + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * k_vec[d];
        }
        scores[static_cast<size_t>(pos)] = dot * scale;
    }

    auto probs = reference_softmax(scores);
    std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
    for (int pos = 0; pos < kv_len; ++pos) {
        const float* v_vec =
            v.data() + static_cast<size_t>(pos) * kv_hidden + kv_head * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out[static_cast<size_t>(d)] += probs[static_cast<size_t>(pos)] * v_vec[d];
        }
    }
    return out;
}

static void fill_kv(std::vector<float>& k, std::vector<float>& v,
                    int num_kv_heads, int head_dim, int seed) {
    const int kv_hidden = num_kv_heads * head_dim;
    const int tokens = static_cast<int>(k.size()) / kv_hidden;
    for (int pos = 0; pos < tokens; ++pos) {
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int idx = pos * kv_hidden + h * head_dim + d;
                k[static_cast<size_t>(idx)] =
                    0.1f * static_cast<float>(seed + pos + 1) +
                    0.2f * static_cast<float>(h + 1) +
                    0.05f * static_cast<float>(d);
                v[static_cast<size_t>(idx)] =
                    static_cast<float>(seed * 10 + 10 * (h + 1) + pos * 3 + d);
            }
        }
    }
}

void test_scheduler_builds_padded_batch() {
    PagedKVCache cache;
    assert(cache.init(1, 2, 2, 2, 8).ok());

    std::vector<float> k0(static_cast<size_t>(5) * 4);
    std::vector<float> v0(k0.size());
    std::vector<float> k1(static_cast<size_t>(3) * 4);
    std::vector<float> v1(k1.size());
    std::vector<float> k2(4);
    std::vector<float> v2(k2.size());
    fill_kv(k0, v0, 2, 2, 1);
    fill_kv(k1, v1, 2, 2, 2);
    fill_kv(k2, v2, 2, 2, 3);

    assert(cache.write_tokens(10, 0, 0, k0.data(), v0.data(), 5).ok());
    assert(cache.write_tokens(20, 0, 0, k1.data(), v1.data(), 3).ok());
    assert(cache.write_tokens(30, 0, 0, k2.data(), v2.data(), 1).ok());

    PagedAttentionScheduler scheduler(cache, 2);
    assert(scheduler.add_sequence(10).ok());
    assert(scheduler.add_sequence(20).ok());
    assert(scheduler.add_sequence(10).ok());
    assert(scheduler.active_sequence_ids().size() == 2);

    auto overflow = scheduler.add_sequence(30);
    assert(!overflow.ok());
    assert(overflow.code() == ErrorCode::OutOfRange);

    auto batch = scheduler.build_batch();
    assert(batch);
    assert(batch->batch_size == 2);
    assert(batch->block_size == 2);
    assert(batch->num_kv_heads == 2);
    assert(batch->head_dim == 2);
    assert(batch->sequence_lengths == std::vector<int>({5, 3}));
    assert(batch->max_blocks_per_sequence == 3);
    assert(batch->block_tables.size() == 6);
    assert(batch->block_tables[2] >= 0);
    assert(batch->block_tables[5] == -1);

    assert(scheduler.remove_sequence(10).ok());
    assert(!scheduler.is_active(10));
    assert(scheduler.is_active(20));

    std::cout << "  PASS test_scheduler_builds_padded_batch\n";
}

void test_scheduler_decode_cpu_multi_sequence() {
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 2;
    const int group_size = num_heads / num_kv_heads;

    PagedKVCache cache;
    assert(cache.init(1, num_kv_heads, head_dim, 2, 8).ok());

    std::vector<float> k0(static_cast<size_t>(5) * num_kv_heads * head_dim);
    std::vector<float> v0(k0.size());
    std::vector<float> k1(static_cast<size_t>(3) * num_kv_heads * head_dim);
    std::vector<float> v1(k1.size());
    fill_kv(k0, v0, num_kv_heads, head_dim, 3);
    fill_kv(k1, v1, num_kv_heads, head_dim, 4);
    assert(cache.write_tokens(10, 0, 0, k0.data(), v0.data(), 5).ok());
    assert(cache.write_tokens(20, 0, 0, k1.data(), v1.data(), 3).ok());

    PagedAttentionScheduler scheduler(cache, 4);
    assert(scheduler.add_sequence(10).ok());
    assert(scheduler.add_sequence(20).ok());

    const std::vector<float> q{
        1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -0.5f, 1.0f,
        0.25f, 1.0f, 1.0f, -0.25f, -1.0f, 0.5f, 0.75f, 0.25f,
    };
    std::vector<float> out(q.size(), 0.0f);
    assert(scheduler.decode_cpu(0, q.data(), out.data(), num_heads, head_dim).ok());

    for (int b = 0; b < 2; ++b) {
        const auto& k = b == 0 ? k0 : k1;
        const auto& v = b == 0 ? v0 : v1;
        const int kv_len = b == 0 ? 5 : 3;
        const float* q_base = q.data() + static_cast<size_t>(b) * num_heads * head_dim;
        const float* out_base = out.data() + static_cast<size_t>(b) * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            const int kv_head = h / group_size;
            auto expected = reference_decode_head(
                q_base + h * head_dim, k, v, kv_len, num_kv_heads, head_dim, kv_head);
            for (int d = 0; d < head_dim; ++d) {
                assert_near(out_base[h * head_dim + d], expected[static_cast<size_t>(d)]);
            }
        }
    }

    std::cout << "  PASS test_scheduler_decode_cpu_multi_sequence\n";
}

int main() {
    std::cout << "test_paged_attention_scheduler:\n";
    test_scheduler_builds_padded_batch();
    test_scheduler_decode_cpu_multi_sequence();
    std::cout << "All tests passed!\n";
    return 0;
}
