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

static std::vector<float> reference_paged_decode_head(
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

void test_paged_kv_cache_write_read_and_free() {
    PagedKVCache cache;
    assert(cache.init(2, 2, 2, 2, 4).ok());

    const int sequence_id = 3;
    const int layer = 1;
    const int token_count = 5;
    const int kv_hidden = 4;
    std::vector<float> k(static_cast<size_t>(token_count) * kv_hidden);
    std::vector<float> v(static_cast<size_t>(token_count) * kv_hidden);
    for (int i = 0; i < token_count * kv_hidden; ++i) {
        k[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        v[static_cast<size_t>(i)] = static_cast<float>(100 + i);
    }

    assert(cache.write_tokens(sequence_id, layer, 0, k.data(), v.data(), token_count).ok());
    assert(cache.has_sequence(sequence_id));
    assert(cache.sequence_length(sequence_id) == token_count);
    assert(cache.sequence_capacity(sequence_id) == 6);
    assert(cache.free_block_count() == 1);

    const auto* table = cache.block_table(sequence_id);
    assert(table != nullptr);
    assert(table->size() == 3);
    assert((*table)[0] != (*table)[1]);
    assert((*table)[1] != (*table)[2]);

    for (int pos = 0; pos < token_count; ++pos) {
        for (int head = 0; head < 2; ++head) {
            auto kp = cache.key_ptr(sequence_id, layer, pos, head);
            auto vp = cache.value_ptr(sequence_id, layer, pos, head);
            assert(kp && vp);
            for (int d = 0; d < 2; ++d) {
                const int flat = pos * kv_hidden + head * 2 + d;
                assert_near((*kp)[d], k[static_cast<size_t>(flat)]);
                assert_near((*vp)[d], v[static_cast<size_t>(flat)]);
            }
        }
    }

    assert(cache.free_sequence(sequence_id).ok());
    assert(!cache.has_sequence(sequence_id));
    assert(cache.free_block_count() == 4);

    std::cout << "  PASS test_paged_kv_cache_write_read_and_free\n";
}

void test_paged_kv_cache_capacity_error() {
    PagedKVCache cache;
    assert(cache.init(1, 1, 2, 2, 2).ok());

    const float k[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const float v[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    auto st = cache.write_tokens(0, 0, 0, k, v, 5);
    assert(!st.ok());
    assert(cache.sequence_length(0) == 0);
    assert(cache.free_block_count() == 2);

    std::cout << "  PASS test_paged_kv_cache_capacity_error\n";
}

void test_paged_attention_decode_matches_contiguous_gqa() {
    const int sequence_id = 7;
    const int layer = 0;
    const int kv_len = 5;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 2;
    const int kv_hidden = num_kv_heads * head_dim;
    const int group_size = num_heads / num_kv_heads;

    PagedKVCache cache;
    assert(cache.init(1, num_kv_heads, head_dim, 2, 4).ok());

    std::vector<float> k(static_cast<size_t>(kv_len) * kv_hidden);
    std::vector<float> v(static_cast<size_t>(kv_len) * kv_hidden);
    for (int pos = 0; pos < kv_len; ++pos) {
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int idx = pos * kv_hidden + h * head_dim + d;
                k[static_cast<size_t>(idx)] =
                    0.1f * static_cast<float>(pos + 1) +
                    0.2f * static_cast<float>(h + 1) +
                    0.05f * static_cast<float>(d);
                v[static_cast<size_t>(idx)] =
                    static_cast<float>(10 * (h + 1) + pos * 3 + d);
            }
        }
    }
    assert(cache.write_tokens(sequence_id, layer, 0, k.data(), v.data(), kv_len).ok());

    const float q[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        -0.5f, 1.0f,
    };
    float out[num_heads * head_dim] = {};
    assert(paged_attention_decode(cache, sequence_id, layer, q, out, num_heads, head_dim).ok());

    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / group_size;
        auto expected = reference_paged_decode_head(
            q + h * head_dim, k, v, kv_len, num_kv_heads, head_dim, kv_head);
        for (int d = 0; d < head_dim; ++d) {
            assert_near(out[h * head_dim + d], expected[static_cast<size_t>(d)]);
        }
    }

    std::cout << "  PASS test_paged_attention_decode_matches_contiguous_gqa\n";
}

int main() {
    std::cout << "test_paged_kv_cache:\n";
    test_paged_kv_cache_write_read_and_free();
    test_paged_kv_cache_capacity_error();
    test_paged_attention_decode_matches_contiguous_gqa();
    std::cout << "All tests passed!\n";
    return 0;
}
