#include <cmath>
#include <cstring>
#include <expected>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/io/bpe_tokenizer.h"
#include "minillm/runtime/sampler.h"
#include "minillm/runtime/paged_kv_cache.h"

using namespace minillm;

namespace {

std::string build_chat_prompt(const std::string& system_prompt,
                              const std::string& user_prompt,
                              bool enable_thinking = false) {
    std::string prompt =
        "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n";
    if (!enable_thinking) {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}

std::expected<MemoryPlan, Status> allocate_runtime_tensors_cpu(
    const Graph& graph,
    RuntimeContext& ctx) {
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

struct PagedGenerationSession {
    int sequence_id;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int max_new_tokens;
};

void fill_input_ids(RuntimeContext& ctx, const Graph& graph, int32_t token) {
    for (const auto& v : graph.values()) {
        if (v.name == "input_ids" && v.kind == ValueKind::Input) {
            auto* t = ctx.get(v.id);
            if (t && t->is_allocated()) {
                auto* ptr = reinterpret_cast<int32_t*>(t->data());
                ptr[0] = token;
            }
        }
    }
}

void fill_input_ids_batch(RuntimeContext& ctx, const Graph& graph,
                          const std::vector<int32_t>& tokens) {
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

const float* get_logits(const RuntimeContext& ctx, ValueId output_id) {
    auto* t = ctx.get(output_id);
    if (!t || !t->is_allocated()) return nullptr;
    return reinterpret_cast<const float*>(t->data());
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [max_tokens]\n";
        std::cerr << "  max_tokens defaults to 32\n";
        return 1;
    }

    std::string gguf_path = argv[1];
    int max_new_tokens = (argc >= 3) ? std::stoi(argv[2]) : 32;

    // --- Step 1: Parse GGUF and extract config ---
    std::cerr << "[1/7] Parsing GGUF file...\n";
    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    auto cfg_result = loader.extract_config(*file_result);
    if (!cfg_result) {
        std::cerr << "Failed to extract config: " << cfg_result.error().to_string() << "\n";
        return 1;
    }
    auto cfg = *cfg_result;

    BPETokenizer tokenizer;
    auto tok_st = tokenizer.init(*file_result);
    if (!tok_st) {
        std::cerr << "Failed to init tokenizer: " << tok_st.error().to_string() << "\n";
        return 1;
    }

    std::cerr << "Config: layers=" << cfg.num_layers << " hidden=" << cfg.hidden_size
              << " heads=" << cfg.num_heads << " kv_heads=" << cfg.num_kv_heads
              << " head_dim=" << cfg.head_dim << " vocab=" << cfg.vocab_size << "\n";

    // --- Step 2: Define multiple concurrent requests ---
    std::vector<std::string> prompts = {
        "Hello",
        "The meaning of life is",
        "Once upon a time",
    };

    std::cerr << "[2/7] Preparing " << prompts.size() << " concurrent requests...\n";

    std::vector<PagedGenerationSession> sessions;
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::string chat_prompt =
            build_chat_prompt("You are a helpful assistant.", prompts[i], false);
        auto encoded = tokenizer.encode(chat_prompt, false, false);
        if (!encoded) {
            std::cerr << "Failed to encode prompt " << i << "\n";
            return 1;
        }
        PagedGenerationSession session;
        session.sequence_id = static_cast<int>(i);
        session.prompt_tokens = *encoded;
        session.max_new_tokens = max_new_tokens;
        sessions.push_back(std::move(session));
    }

    // --- Step 3: Build decode graph (shared across sequences) ---
    std::cerr << "[3/7] Building decode graph...\n";

    TransformerConfig decode_cfg = cfg;
    decode_cfg.seq_len = 1;

    Graph decode_graph;
    GraphBuilder decode_gb(decode_graph);
    TransformerGraphBuilder decode_tb(decode_gb);
    auto decode_result = decode_tb.build_transformer(decode_cfg);
    if (!decode_result) {
        std::cerr << "Failed to build decode graph: " << decode_result.error().to_string() << "\n";
        return 1;
    }
    auto st = decode_graph.validate();
    if (!st.ok()) {
        std::cerr << "Decode graph validation failed: " << st.to_string() << "\n";
        return 1;
    }

    // --- Step 4: Initialize PagedKVCache ---
    std::cerr << "[4/7] Initializing PagedKVCache...\n";

    const int block_size = 16;
    const int max_blocks = 256;
    PagedKVCache paged_cache;
    st = paged_cache.init(
        static_cast<int>(cfg.num_layers),
        static_cast<int>(cfg.num_kv_heads),
        static_cast<int>(cfg.head_dim),
        block_size,
        max_blocks);
    if (!st.ok()) {
        std::cerr << "Failed to init PagedKVCache: " << st.to_string() << "\n";
        return 1;
    }

    for (const auto& session : sessions) {
        const int total_tokens = static_cast<int>(session.prompt_tokens.size()) + session.max_new_tokens;
        st = paged_cache.ensure_sequence(session.sequence_id);
        if (!st.ok()) {
            std::cerr << "Failed to ensure sequence: " << st.to_string() << "\n";
            return 1;
        }
        st = paged_cache.reserve_sequence(session.sequence_id, total_tokens);
        if (!st.ok()) {
            std::cerr << "Failed to reserve sequence blocks: " << st.to_string() << "\n";
            return 1;
        }
    }

    // --- Step 5: Load shared weights and build shared decode context ---
    std::cerr << "[5/7] Loading shared weights...\n";

    // Use decode_graph for weight loading (same architecture as prefill)
    TransformerConfig max_prefill_cfg = cfg;
    int max_prompt_len = 0;
    for (const auto& s : sessions) {
        max_prompt_len = std::max(max_prompt_len, static_cast<int>(s.prompt_tokens.size()));
    }
    max_prefill_cfg.seq_len = max_prompt_len;
    Graph weight_graph;
    GraphBuilder weight_gb(weight_graph);
    TransformerGraphBuilder weight_tb(weight_gb);
    auto weight_result = weight_tb.build_transformer(max_prefill_cfg);
    if (!weight_result) {
        std::cerr << "Failed to build weight graph: " << weight_result.error().to_string() << "\n";
        return 1;
    }

    auto shared_weights = loader.load_shared_weights(*file_result, weight_graph);
    if (!shared_weights) {
        std::cerr << "Failed to load shared weights: "
                  << shared_weights.error().to_string() << "\n";
        return 1;
    }

    // Shared decode context and executor (reused across sequences)
    RuntimeContext decode_ctx;
    auto decode_plan = allocate_runtime_tensors_cpu(decode_graph, decode_ctx);
    if (!decode_plan) {
        std::cerr << "Failed to allocate decode tensors: "
                  << decode_plan.error().to_string() << "\n";
        return 1;
    }
    st = shared_weights->bind(decode_graph, decode_ctx);
    if (!st.ok()) {
        std::cerr << "Failed to bind decode weights: " << st.to_string() << "\n";
        return 1;
    }

    KernelRegistry registry;
    register_cpu_kernels(registry);
    auto backend = std::make_shared<CpuBackend>();

    CpuExecutor decode_exec(backend, registry);
    st = decode_exec.compile(decode_graph);
    if (!st.ok()) {
        std::cerr << "Decode compile failed: " << st.to_string() << "\n";
        return 1;
    }

    SamplingConfig samp_cfg;
    // Match the llama.cpp comparison path: deterministic greedy decode.
    samp_cfg.greedy = true;
    samp_cfg.temperature = 0.0f;
    samp_cfg.top_k = 1;
    samp_cfg.top_p = 1.0f;
    samp_cfg.repetition_penalty = 1.1f;
    Sampler sampler(42);

    const int num_layers = static_cast<int>(cfg.num_layers);
    const int num_kv_heads = static_cast<int>(cfg.num_kv_heads);
    const int head_dim = static_cast<int>(cfg.head_dim);
    const int kv_hidden = num_kv_heads * head_dim;
    const int vocab_size = static_cast<int>(cfg.vocab_size);

    // --- Step 6: Prefill each sequence using contiguous KV, then copy to paged ---
    std::cerr << "[6/7] Prefilling sequences with contiguous KV cache...\n";

    for (auto& session : sessions) {
        int prompt_len = static_cast<int>(session.prompt_tokens.size());

        TransformerConfig seq_prefill_cfg = cfg;
        seq_prefill_cfg.seq_len = prompt_len;

        Graph seq_prefill_graph;
        GraphBuilder seq_gb(seq_prefill_graph);
        TransformerGraphBuilder seq_tb(seq_gb);
        auto seq_result = seq_tb.build_transformer(seq_prefill_cfg);
        if (!seq_result) {
            std::cerr << "Failed to build seq prefill graph: " << seq_result.error().to_string() << "\n";
            return 1;
        }

        RuntimeContext seq_ctx;
        for (const auto& v : seq_prefill_graph.values()) {
            if (v.kind != ValueKind::Input) continue;
            auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
            auto s2 = t->allocate_cpu();
            if (!s2.ok()) { std::cerr << s2.to_string() << "\n"; return 1; }
            s2 = seq_ctx.emplace(v.id, std::move(t));
            if (!s2.ok()) { std::cerr << s2.to_string() << "\n"; return 1; }
        }
        auto seq_plan = seq_ctx.allocate_intermediates_planned(seq_prefill_graph);
        if (!seq_plan) {
            std::cerr << "Failed to allocate seq prefill intermediates: "
                      << seq_plan.error().to_string() << "\n";
            return 1;
        }
        st = shared_weights->bind(seq_prefill_graph, seq_ctx);
        if (!st.ok()) {
            std::cerr << "Failed to bind weights: " << st.to_string() << "\n";
            return 1;
        }

        auto kv_cache = std::make_shared<KVCache>();
        kv_cache->init(num_layers, num_kv_heads, head_dim,
                       prompt_len + session.max_new_tokens);

        seq_ctx.set_kv_cache(kv_cache);
        seq_ctx.set_kv_cache_advance_tokens(prompt_len);

        CpuExecutor seq_exec(backend, registry);
        st = seq_exec.compile(seq_prefill_graph);
        if (!st.ok()) {
            std::cerr << "Seq compile failed: " << st.to_string() << "\n";
            return 1;
        }

        fill_input_ids_batch(seq_ctx, seq_prefill_graph, session.prompt_tokens);
        st = seq_exec.run(seq_ctx);
        if (!st.ok()) {
            std::cerr << "Seq prefill failed: " << st.to_string() << "\n";
            return 1;
        }

        // Copy KV from contiguous cache to paged cache
        for (int layer = 0; layer < num_layers; ++layer) {
            const float* src_k = kv_cache->k_data(layer);
            const float* src_v = kv_cache->v_data(layer);
            st = paged_cache.write_tokens(session.sequence_id, layer, 0,
                                          src_k, src_v, prompt_len);
            if (!st.ok()) {
                std::cerr << "Failed to write paged KV: " << st.to_string() << "\n";
                return 1;
            }
        }
        st = paged_cache.set_sequence_length(session.sequence_id, prompt_len);
        if (!st.ok()) {
            std::cerr << "Failed to set paged seq length: " << st.to_string() << "\n";
            return 1;
        }

        // Sample first token from prefill output
        auto* logits_tensor = seq_ctx.get(*seq_result);
        if (!logits_tensor) {
            std::cerr << "Output tensor not found for seq " << session.sequence_id << "\n";
            return 1;
        }
        auto* logits_data = reinterpret_cast<const float*>(logits_tensor->data());
        const float* last_logits = logits_data + (prompt_len - 1) * vocab_size;

        int32_t first_token = sampler.sample(
            last_logits, vocab_size, samp_cfg, session.prompt_tokens);

        session.generated_tokens.push_back(first_token);

        std::cerr << "  Seq " << session.sequence_id
                  << " prefill -> token " << first_token << "\n";
    }

    // --- Step 7: Decode loop using paged attention ---
    // The attention kernel reads K/V from the paged cache and writes newly
    // projected K/V directly into it, eliminating the need for per-sequence
    // contiguous KV caches.
    std::cerr << "[7/7] Decode loop for " << sessions.size() << " sequences"
              << " (paged attention)...\n\n";

    const int32_t eos_id = tokenizer.eos_token_id();
    const int32_t think_start_id = 151667;
    const int32_t think_end_id = 151668;
    std::vector<bool> seq_done(sessions.size(), false);
    int active_count = static_cast<int>(sessions.size());

    // Per-sequence state: just track the current KV position
    std::vector<int> seq_positions(sessions.size());
    for (size_t si = 0; si < sessions.size(); ++si) {
        seq_positions[si] = static_cast<int>(sessions[si].prompt_tokens.size());
    }

    for (int step = 0; step < max_new_tokens && active_count > 0; ++step) {
        for (size_t si = 0; si < sessions.size(); ++si) {
            if (seq_done[si]) continue;
            auto& session = sessions[si];

            // Configure paged attention for this sequence
            decode_ctx.set_paged_kv_cache(&paged_cache);
            decode_ctx.set_paged_sequence_id(session.sequence_id);

            fill_input_ids(decode_ctx, decode_graph, session.generated_tokens.back());

            st = decode_exec.run(decode_ctx);
            if (!st.ok()) {
                std::cerr << "Decode failed for seq " << session.sequence_id
                          << " step " << step << ": " << st.to_string() << "\n";
                seq_done[si] = true;
                active_count--;
                continue;
            }

            seq_positions[si]++;

            // Get logits and sample
            auto* logits = get_logits(decode_ctx, *decode_result);
            if (!logits) {
                std::cerr << "No logits for seq " << session.sequence_id << "\n";
                seq_done[si] = true;
                active_count--;
                continue;
            }

            auto history = session.prompt_tokens;
            history.insert(history.end(), session.generated_tokens.begin(), session.generated_tokens.end());
            int32_t next_token = sampler.sample(
                logits, vocab_size, samp_cfg, history);

            if (next_token == eos_id) {
                seq_done[si] = true;
                active_count--;
                continue;
            }

            session.generated_tokens.push_back(next_token);

            if (static_cast<int>(session.generated_tokens.size()) >= session.max_new_tokens) {
                seq_done[si] = true;
                active_count--;
            }
        }
    }

    // Print results
    std::cerr << "\n=== Generation Results ===\n\n";
    for (size_t si = 0; si < sessions.size(); ++si) {
        const auto& session = sessions[si];
        std::cerr << "Seq " << session.sequence_id << " (prompt: \""
                  << prompts[si] << "\")\n";
        std::cerr << "  Prompt tokens: " << session.prompt_tokens.size() << "\n";
        std::cerr << "  Generated tokens: " << session.generated_tokens.size() << "\n";

        std::cout << "Prompt: " << prompts[si] << "\n";
        std::cout << "Response: ";
        for (int32_t tok : session.generated_tokens) {
            if (tok == think_start_id || tok == think_end_id) continue;
            auto decoded = tokenizer.decode({tok});
            if (decoded) std::cout << *decoded;
        }
        std::cout << "\n\n";
    }

    // Print paged cache stats
    std::cerr << "PagedKVCache stats:\n";
    std::cerr << "  Block size: " << paged_cache.block_size() << "\n";
    std::cerr << "  Total blocks: " << paged_cache.max_blocks() << "\n";
    std::cerr << "  Free blocks: " << paged_cache.free_block_count() << "\n";
    std::cerr << "  Used blocks: " << (paged_cache.max_blocks() - paged_cache.free_block_count()) << "\n";

    return 0;
}
