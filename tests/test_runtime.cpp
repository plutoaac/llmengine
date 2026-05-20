#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "minillm/minillm.h"

using namespace minillm;

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

static std::vector<float> reference_attention_row(
    const float* q, const float* k, const float* v,
    int kv_len, int head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(kv_len));
    for (int i = 0; i < kv_len; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * k[i * head_dim + d];
        }
        scores[static_cast<size_t>(i)] = dot * scale;
    }

    auto probs = reference_softmax(scores);
    std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
    for (int i = 0; i < kv_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            out[static_cast<size_t>(d)] +=
                probs[static_cast<size_t>(i)] * v[i * head_dim + d];
        }
    }
    return out;
}

void test_sampler_greedy() {
    float logits[] = {0.0f, 3.0f, 1.0f, -2.0f};
    SamplingConfig cfg;
    cfg.greedy = true;
    cfg.temperature = 0.0f;

    Sampler sampler(123);
    assert(sampler.sample(logits, 4, cfg) == 1);
    std::cout << "  PASS test_sampler_greedy\n";
}

void test_sampler_top_tokens() {
    float logits[] = {0.0f, 3.0f, 1.0f, -2.0f};
    SamplingConfig cfg;

    Sampler sampler(123);
    auto top = sampler.get_top_tokens(logits, 4, cfg, 2);

    assert(top.size() == 2);
    assert(top[0].first == 1);
    assert(top[1].first == 2);
    assert(top[0].second > top[1].second);
    std::cout << "  PASS test_sampler_top_tokens\n";
}

void test_sampler_invalid_inputs() {
    SamplingConfig cfg;
    Sampler sampler(123);
    float logits[] = {0.0f, 1.0f, 2.0f, 3.0f};

    assert(sampler.sample(nullptr, 0, cfg) == -1);
    assert(sampler.get_top_tokens(nullptr, 0, cfg, 1).empty());
    assert(sampler.get_top_tokens(logits, 4, cfg, 0).empty());
    std::cout << "  PASS test_sampler_invalid_inputs\n";
}

void test_kv_cache_bounds() {
    KVCache cache;
    cache.init(2, 4, 8, 16);

    assert(cache.initialized());
    assert(cache.num_layers() == 2);
    assert(cache.kv_hidden() == 32);
    assert(cache.cached_len() == 0);
    assert(cache.can_append(16));
    assert(!cache.can_append(17));

    cache.advance(3);
    assert(cache.cached_len() == 3);
    assert(cache.can_append(13));
    assert(!cache.can_append(14));

    cache.set_cached_len(100);
    assert(cache.cached_len() == 16);
    assert(!cache.can_append(1));

    cache.k_data(0)[0] = 42.0f;
    cache.v_data(1)[cache.kv_hidden() - 1] = 7.0f;
    cache.reset();
    assert(cache.cached_len() == 0);
    assert(std::abs(cache.k_data(0)[0]) < 1e-6f);
    assert(std::abs(cache.v_data(1)[cache.kv_hidden() - 1]) < 1e-6f);

    std::cout << "  PASS test_kv_cache_bounds\n";
}

void test_kv_cache_executor_advance_and_decode_attention() {
    auto cache = std::make_shared<KVCache>();
    cache->init(1, 1, 2, 4);

    KernelRegistry registry;
    register_cpu_kernels(registry);
    auto backend = std::make_shared<CpuBackend>();

    Graph prefill_graph;
    GraphBuilder prefill_gb(prefill_graph);
    auto prefill_q = prefill_gb.input("q", Shape({1, 2, 4}), DType::Float32);
    auto prefill_k = prefill_gb.input("k", Shape({1, 2, 2}), DType::Float32);
    auto prefill_v = prefill_gb.input("v", Shape({1, 2, 2}), DType::Float32);
    assert(prefill_q && prefill_k && prefill_v);
    auto prefill_out = prefill_gb.attention(
        *prefill_q, *prefill_k, *prefill_v, true,
        2, 1, 2, 0, "prefill_attn");
    assert(prefill_out);

    Tensor pq("q", Shape({1, 2, 4}), DType::Float32);
    Tensor pk("k", Shape({1, 2, 2}), DType::Float32);
    Tensor pv("v", Shape({1, 2, 2}), DType::Float32);
    assert(pq.allocate_cpu().ok());
    assert(pk.allocate_cpu().ok());
    assert(pv.allocate_cpu().ok());

    float* pq_data = reinterpret_cast<float*>(pq.data());
    pq_data[0] = 1.0f; pq_data[1] = 0.0f; pq_data[2] = 0.0f; pq_data[3] = 1.0f;
    pq_data[4] = 0.5f; pq_data[5] = 0.5f; pq_data[6] = 1.0f; pq_data[7] = 1.0f;

    float* pk_data = reinterpret_cast<float*>(pk.data());
    pk_data[0] = 1.0f; pk_data[1] = 0.0f;
    pk_data[2] = 0.0f; pk_data[3] = 1.0f;

    float* pv_data = reinterpret_cast<float*>(pv.data());
    pv_data[0] = 10.0f; pv_data[1] = 0.0f;
    pv_data[2] = 0.0f; pv_data[3] = 20.0f;

    RuntimeContext prefill_ctx;
    assert(prefill_ctx.bind(*prefill_q, &pq).ok());
    assert(prefill_ctx.bind(*prefill_k, &pk).ok());
    assert(prefill_ctx.bind(*prefill_v, &pv).ok());
    assert(prefill_ctx.allocate_intermediates(prefill_graph).ok());
    prefill_ctx.set_kv_cache(cache);
    prefill_ctx.set_kv_cache_advance_tokens(2);

    CpuExecutor prefill_exec(backend, registry);
    assert(prefill_exec.compile(prefill_graph).ok());
    assert(prefill_exec.run(prefill_ctx).ok());
    assert(cache->cached_len() == 2);

    Graph decode_graph;
    GraphBuilder decode_gb(decode_graph);
    auto decode_q = decode_gb.input("q", Shape({1, 1, 4}), DType::Float32);
    auto decode_k = decode_gb.input("k", Shape({1, 1, 2}), DType::Float32);
    auto decode_v = decode_gb.input("v", Shape({1, 1, 2}), DType::Float32);
    assert(decode_q && decode_k && decode_v);
    auto decode_out = decode_gb.attention(
        *decode_q, *decode_k, *decode_v, true,
        2, 1, 2, 0, "decode_attn");
    assert(decode_out);

    Tensor dq("q", Shape({1, 1, 4}), DType::Float32);
    Tensor dk("k", Shape({1, 1, 2}), DType::Float32);
    Tensor dv("v", Shape({1, 1, 2}), DType::Float32);
    assert(dq.allocate_cpu().ok());
    assert(dk.allocate_cpu().ok());
    assert(dv.allocate_cpu().ok());

    float* dq_data = reinterpret_cast<float*>(dq.data());
    dq_data[0] = 1.0f; dq_data[1] = 0.0f;
    dq_data[2] = 0.0f; dq_data[3] = 1.0f;
    float* dk_data = reinterpret_cast<float*>(dk.data());
    dk_data[0] = 1.0f; dk_data[1] = 1.0f;
    float* dv_data = reinterpret_cast<float*>(dv.data());
    dv_data[0] = 30.0f; dv_data[1] = 40.0f;

    RuntimeContext decode_ctx;
    assert(decode_ctx.bind(*decode_q, &dq).ok());
    assert(decode_ctx.bind(*decode_k, &dk).ok());
    assert(decode_ctx.bind(*decode_v, &dv).ok());
    assert(decode_ctx.allocate_intermediates(decode_graph).ok());
    decode_ctx.set_kv_cache(cache);
    decode_ctx.set_kv_cache_advance_tokens(1);

    CpuExecutor decode_exec(backend, registry);
    assert(decode_exec.compile(decode_graph).ok());
    assert(decode_exec.run(decode_ctx).ok());
    assert(cache->cached_len() == 3);

    Tensor* out_tensor = decode_ctx.get(*decode_out);
    assert(out_tensor != nullptr);
    const float* result = reinterpret_cast<const float*>(out_tensor->data());
    const float* cache_k = cache->k_data(0);
    const float* cache_v = cache->v_data(0);
    auto expected_h0 = reference_attention_row(dq_data, cache_k, cache_v, 3, 2);
    auto expected_h1 = reference_attention_row(dq_data + 2, cache_k, cache_v, 3, 2);

    assert(std::abs(result[0] - expected_h0[0]) < 1e-5f);
    assert(std::abs(result[1] - expected_h0[1]) < 1e-5f);
    assert(std::abs(result[2] - expected_h1[0]) < 1e-5f);
    assert(std::abs(result[3] - expected_h1[1]) < 1e-5f);

    std::cout << "  PASS test_kv_cache_executor_advance_and_decode_attention\n";
}

void test_cpu_executor_softmax_transpose_reshape() {
    Graph g;
    GraphBuilder gb(g);

    auto x = gb.input("x", Shape({2, 3}), DType::Float32);
    assert(x);
    auto probs = gb.softmax(*x, -1, "softmax");
    assert(probs);
    auto transposed = gb.transpose(*probs, 0, 1, "transpose");
    assert(transposed);
    auto flat = gb.reshape(*transposed, Shape({6}), "reshape");
    assert(flat);
    auto out = gb.output(*flat, "out");
    assert(out);

    Tensor x_tensor("x", Shape({2, 3}), DType::Float32);
    assert(x_tensor.allocate_cpu().ok());
    float* x_data = reinterpret_cast<float*>(x_tensor.data());
    x_data[0] = 1.0f;
    x_data[1] = 2.0f;
    x_data[2] = 3.0f;
    x_data[3] = 1.0f;
    x_data[4] = 1.0f;
    x_data[5] = 1.0f;

    RuntimeContext ctx;
    assert(ctx.bind(*x, &x_tensor).ok());
    assert(ctx.allocate_intermediates(g).ok());

    KernelRegistry registry;
    register_cpu_kernels(registry);
    CpuExecutor executor(std::make_shared<CpuBackend>(), registry);
    assert(executor.compile(g).ok());
    assert(executor.run(ctx).ok());

    Tensor* out_tensor = ctx.get(*flat);
    assert(out_tensor != nullptr);
    const float* y = reinterpret_cast<const float*>(out_tensor->data());

    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(2.0f);
    const float e3 = std::exp(3.0f);
    const float sum = e1 + e2 + e3;
    const float expected[] = {
        e1 / sum, 1.0f / 3.0f,
        e2 / sum, 1.0f / 3.0f,
        e3 / sum, 1.0f / 3.0f,
    };
    for (int i = 0; i < 6; ++i) {
        assert(std::abs(y[i] - expected[i]) < 1e-5f);
    }

    std::cout << "  PASS test_cpu_executor_softmax_transpose_reshape\n";
}

void test_cpu_executor_linear_bias() {
    Graph g;
    GraphBuilder gb(g);

    auto x = gb.input("x", Shape({1, 2, 3}), DType::Float32);
    assert(x);
    auto w = gb.constant("w", Shape({2, 3}), DType::Float32);
    assert(w);
    auto b = gb.constant("b", Shape({2}), DType::Float32);
    assert(b);
    auto y = gb.linear(*x, *w, *b, "linear");
    assert(y);
    auto out = gb.output(*y, "out");
    assert(out);

    Tensor x_tensor("x", Shape({1, 2, 3}), DType::Float32);
    Tensor w_tensor("w", Shape({2, 3}), DType::Float32);
    Tensor b_tensor("b", Shape({2}), DType::Float32);
    assert(x_tensor.allocate_cpu().ok());
    assert(w_tensor.allocate_cpu().ok());
    assert(b_tensor.allocate_cpu().ok());

    float* x_data = reinterpret_cast<float*>(x_tensor.data());
    x_data[0] = 1.0f; x_data[1] = 2.0f; x_data[2] = 3.0f;
    x_data[3] = 4.0f; x_data[4] = 5.0f; x_data[5] = 6.0f;

    float* w_data = reinterpret_cast<float*>(w_tensor.data());
    w_data[0] = 1.0f; w_data[1] = 0.0f; w_data[2] = 1.0f;
    w_data[3] = 0.0f; w_data[4] = 1.0f; w_data[5] = 1.0f;

    float* b_data = reinterpret_cast<float*>(b_tensor.data());
    b_data[0] = 0.5f;
    b_data[1] = -1.0f;

    RuntimeContext ctx;
    assert(ctx.bind(*x, &x_tensor).ok());
    assert(ctx.bind(*w, &w_tensor).ok());
    assert(ctx.bind(*b, &b_tensor).ok());
    assert(ctx.allocate_intermediates(g).ok());

    KernelRegistry registry;
    register_cpu_kernels(registry);
    CpuExecutor executor(std::make_shared<CpuBackend>(), registry);
    assert(executor.compile(g).ok());
    assert(executor.run(ctx).ok());

    Tensor* out_tensor = ctx.get(*y);
    assert(out_tensor != nullptr);
    const float* result = reinterpret_cast<const float*>(out_tensor->data());
    const float expected[] = {4.5f, 4.0f, 10.5f, 10.0f};
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(result[i] - expected[i]) < 1e-5f);
    }

    std::cout << "  PASS test_cpu_executor_linear_bias\n";
}

void test_cpu_executor_embedding_rank1() {
    Graph g;
    GraphBuilder gb(g);

    auto ids = gb.input("ids", Shape({3}), DType::Int32);
    assert(ids);
    auto w = gb.constant("w", Shape({4, 2}), DType::Float32);
    assert(w);
    auto y = gb.embedding(*ids, *w, "embedding");
    assert(y);
    auto out = gb.output(*y, "out");
    assert(out);

    Tensor ids_tensor("ids", Shape({3}), DType::Int32);
    Tensor w_tensor("w", Shape({4, 2}), DType::Float32);
    assert(ids_tensor.allocate_cpu().ok());
    assert(w_tensor.allocate_cpu().ok());

    int* ids_data = reinterpret_cast<int*>(ids_tensor.data());
    ids_data[0] = 2;
    ids_data[1] = 0;
    ids_data[2] = 3;

    float* w_data = reinterpret_cast<float*>(w_tensor.data());
    for (int i = 0; i < 8; ++i) {
        w_data[i] = static_cast<float>(i + 1);
    }

    RuntimeContext ctx;
    assert(ctx.bind(*ids, &ids_tensor).ok());
    assert(ctx.bind(*w, &w_tensor).ok());
    assert(ctx.allocate_intermediates(g).ok());

    KernelRegistry registry;
    register_cpu_kernels(registry);
    CpuExecutor executor(std::make_shared<CpuBackend>(), registry);
    assert(executor.compile(g).ok());
    assert(executor.run(ctx).ok());

    Tensor* out_tensor = ctx.get(*y);
    assert(out_tensor != nullptr);
    const float* result = reinterpret_cast<const float*>(out_tensor->data());
    const float expected[] = {5.0f, 6.0f, 1.0f, 2.0f, 7.0f, 8.0f};
    for (int i = 0; i < 6; ++i) {
        assert(std::abs(result[i] - expected[i]) < 1e-6f);
    }

    std::cout << "  PASS test_cpu_executor_embedding_rank1\n";
}

void test_cpu_executor_embedding_id_out_of_range() {
    Graph g;
    GraphBuilder gb(g);

    auto ids = gb.input("ids", Shape({1}), DType::Int32);
    assert(ids);
    auto w = gb.constant("w", Shape({2, 2}), DType::Float32);
    assert(w);
    auto y = gb.embedding(*ids, *w, "embedding");
    assert(y);

    Tensor ids_tensor("ids", Shape({1}), DType::Int32);
    Tensor w_tensor("w", Shape({2, 2}), DType::Float32);
    assert(ids_tensor.allocate_cpu().ok());
    assert(w_tensor.allocate_cpu().ok());
    reinterpret_cast<int*>(ids_tensor.data())[0] = 2;

    RuntimeContext ctx;
    assert(ctx.bind(*ids, &ids_tensor).ok());
    assert(ctx.bind(*w, &w_tensor).ok());
    assert(ctx.allocate_intermediates(g).ok());

    KernelRegistry registry;
    register_cpu_kernels(registry);
    CpuExecutor executor(std::make_shared<CpuBackend>(), registry);
    assert(executor.compile(g).ok());
    auto st = executor.run(ctx);
    assert(!st.ok());
    assert(st.code() == ErrorCode::RuntimeError);

    std::cout << "  PASS test_cpu_executor_embedding_id_out_of_range\n";
}

int main() {
    std::cout << "test_runtime:\n";
    test_sampler_greedy();
    test_sampler_top_tokens();
    test_sampler_invalid_inputs();
    test_kv_cache_bounds();
    test_kv_cache_executor_advance_and_decode_attention();
    test_cpu_executor_softmax_transpose_reshape();
    test_cpu_executor_linear_bias();
    test_cpu_executor_embedding_rank1();
    test_cpu_executor_embedding_id_out_of_range();
    std::cout << "All tests passed!\n";
    return 0;
}
