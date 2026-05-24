#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/runtime/paged_kv_cache.h"
#include "minillm/runtime/paged_attention_scheduler.h"

using namespace minillm;

namespace {

void fill_input_ids(RuntimeContext& ctx, const Graph& graph, const std::vector<int32_t>& tokens) {
    for (const auto& v : graph.values()) {
        if (v.name == "input_ids" && v.kind == ValueKind::Input) {
            auto* t = ctx.get(v.id);
            if (t && t->is_allocated()) {
                auto* ptr = reinterpret_cast<int32_t*>(t->data());
                for (size_t i = 0; i < static_cast<size_t>(tokens.size()); ++i) {
                    ptr[i] = tokens[i];
                }
            }
        }
    }
}

std::expected<MemoryPlan, Status> allocate_runtime_tensors(
    const Graph& graph, RuntimeContext& ctx) {
    for (const auto& v : graph.values()) {
        if (v.kind != ValueKind::Input) continue;
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        auto st = t->allocate_cpu();
        if (!st.ok()) return std::unexpected(st);
        st = ctx.emplace(v.id, std::move(t));
        if (!st.ok()) return std::unexpected(st);
    }
    return ctx.allocate_intermediates_planned(graph);
}

float cosine_similarity(const float* a, const float* b, int n) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0 ? dot / denom : 0.0f;
}

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

void test_contiguous_vs_paged_kv() {
    std::cout << "  test_contiguous_vs_paged_kv: comparing contiguous KV cache and paged KV cache outputs...\n";

    constexpr int num_layers = 2;
    constexpr int num_heads = 4;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 32;
    constexpr int kv_hidden = num_kv_heads * head_dim;
    constexpr int hidden_size = num_heads * head_dim;
    constexpr int seq_len = 16;
    constexpr int block_size = 8;
    constexpr int max_blocks = 32;
    constexpr int max_seq_len = 64;

    auto shared_weights = std::make_shared<KVCache>();
    shared_weights->init(num_layers, num_kv_heads, head_dim, max_seq_len);

    PagedKVCache paged_cache;
    auto st = paged_cache.init(num_layers, num_kv_heads, head_dim, block_size, max_blocks);
    assert(st.ok());
    st = paged_cache.ensure_sequence(0);
    assert(st.ok());
    st = paged_cache.reserve_sequence(0, max_seq_len);
    assert(st.ok());

    for (int layer = 0; layer < num_layers; ++layer) {
        float* cont_k = shared_weights->k_data(layer);
        float* cont_v = shared_weights->v_data(layer);

        std::vector<float> k_row(kv_hidden);
        std::vector<float> v_row(kv_hidden);
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int j = 0; j < kv_hidden; ++j) {
                k_row[j] = 0.01f * static_cast<float>(pos * kv_hidden + j + 1);
                v_row[j] = 0.02f * static_cast<float>(pos * kv_hidden + j + 1);
            }
            std::memcpy(cont_k + pos * kv_hidden, k_row.data(), kv_hidden * sizeof(float));
            std::memcpy(cont_v + pos * kv_hidden, v_row.data(), kv_hidden * sizeof(float));

            st = paged_cache.write_tokens(0, layer, pos, k_row.data(), v_row.data(), 1);
            assert(st.ok());
        }
    }
    shared_weights->set_cached_len(seq_len);
    st = paged_cache.set_sequence_length(0, seq_len);
    assert(st.ok());

    std::vector<float> q_buf(static_cast<size_t>(num_heads) * head_dim);
    for (int i = 0; i < num_heads * head_dim; ++i) {
        q_buf[i] = 0.001f * static_cast<float>(i + 1);
    }

    std::vector<float> cont_output(static_cast<size_t>(num_heads) * head_dim);
    std::vector<float> paged_output(static_cast<size_t>(num_heads) * head_dim);

    for (int layer = 0; layer < num_layers; ++layer) {
        cpu::flash_sdpa_decode(q_buf.data(),
                                shared_weights->k_data(layer),
                                shared_weights->v_data(layer),
                                cont_output.data(),
                                num_heads, num_kv_heads, head_dim, seq_len);

        st = paged_attention_decode(paged_cache, 0, layer,
                                     q_buf.data(), paged_output.data(),
                                     num_heads, head_dim);
        assert(st.ok());

        float cos_sim = cosine_similarity(cont_output.data(), paged_output.data(),
                                           num_heads * head_dim);
        float max_diff = max_abs_diff(cont_output.data(), paged_output.data(),
                                      num_heads * head_dim);

        std::cout << "    Layer " << layer
                  << ": cosine_sim=" << cos_sim
                  << " max_abs_diff=" << max_diff << "\n";

        assert(cos_sim > 0.999f);
        assert(max_diff < 1e-4f);
    }

    std::cout << "    PASSED\n";
}

void test_multisequence_paged_decode() {
    std::cout << "  test_multisequence_paged_decode: multiple sequences in paged cache...\n";

    constexpr int num_layers = 2;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 32;
    constexpr int block_size = 8;
    constexpr int max_blocks = 32;
    constexpr int kv_hidden = num_kv_heads * head_dim;

    PagedKVCache cache;
    auto st = cache.init(num_layers, num_kv_heads, head_dim, block_size, max_blocks);
    assert(st.ok());

    int free_before = cache.free_block_count();

    for (int seq = 0; seq < 3; ++seq) {
        st = cache.ensure_sequence(seq);
        assert(st.ok());
        st = cache.reserve_sequence(seq, 16);
        assert(st.ok());

        std::vector<float> k_row(kv_hidden);
        std::vector<float> v_row(kv_hidden);
        for (int pos = 0; pos < 10; ++pos) {
            for (int j = 0; j < kv_hidden; ++j) {
                k_row[j] = 0.01f * static_cast<float>((seq + 1) * (pos * kv_hidden + j + 1));
                v_row[j] = 0.02f * static_cast<float>((seq + 1) * (pos * kv_hidden + j + 1));
            }
            st = cache.write_tokens(seq, 0, pos, k_row.data(), v_row.data(), 1);
            assert(st.ok());
        }
    }

    int free_after_reserve = cache.free_block_count();
    assert(free_after_reserve < free_before);

    PagedAttentionScheduler scheduler(cache, 3);
    for (int seq = 0; seq < 3; ++seq) {
        st = scheduler.add_sequence(seq);
        assert(st.ok());
    }

    auto batch = scheduler.build_batch();
    assert(batch.has_value());
    assert(batch->batch_size == 3);

    int num_heads = 4;
    std::vector<float> q_data(static_cast<size_t>(3) * num_heads * head_dim, 0.01f);
    std::vector<float> output_data(static_cast<size_t>(3) * num_heads * head_dim, 0.0f);

    st = scheduler.decode_cpu(0, q_data.data(), output_data.data(), num_heads, head_dim);
    assert(st.ok());

    for (int seq = 0; seq < 3; ++seq) {
        st = cache.free_sequence(seq);
        assert(st.ok());
    }

    int free_after_free = cache.free_block_count();
    assert(free_after_free == free_before);

    std::cout << "    PASSED\n";
}

void test_block_reuse() {
    std::cout << "  test_block_reuse: freed blocks are reused by new sequences...\n";

    constexpr int num_layers = 1;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 16;
    constexpr int block_size = 4;
    constexpr int max_blocks = 16;

    PagedKVCache cache;
    auto st = cache.init(num_layers, num_kv_heads, head_dim, block_size, max_blocks);
    assert(st.ok());

    int initial_free = cache.free_block_count();

    st = cache.ensure_sequence(0);
    assert(st.ok());
    st = cache.reserve_sequence(0, 8);
    assert(st.ok());

    assert(cache.free_block_count() == initial_free - 2);

    st = cache.free_sequence(0);
    assert(st.ok());

    assert(cache.free_block_count() == initial_free);

    st = cache.ensure_sequence(1);
    assert(st.ok());
    st = cache.reserve_sequence(1, 8);
    assert(st.ok());

    assert(cache.free_block_count() == initial_free - 2);

    std::cout << "    PASSED\n";
}

} // namespace

int main() {
    std::cout << "E2E verification tests:\n";
    test_contiguous_vs_paged_kv();
    test_multisequence_paged_decode();
    test_block_reuse();
    std::cout << "All E2E verification tests passed!\n";
    return 0;
}