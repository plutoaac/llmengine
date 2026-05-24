#include <cmath>
#include <expected>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/io/bpe_tokenizer.h"
#include "minillm/runtime/sampler.h"

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

void print_plan_summary(const char* label, const MemoryPlan& plan) {
    constexpr double MiB = 1024.0 * 1024.0;
    std::cerr << label << " activation arena: "
              << (static_cast<double>(plan.planned_bytes) / MiB)
              << " MiB planned vs "
              << (static_cast<double>(plan.naive_bytes) / MiB)
              << " MiB naive, reuse saving "
              << (plan.savings_ratio() * 100.0) << "%\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [prompt] [max_tokens]\n";
        std::cerr << "  prompt defaults to \"Hello\"\n";
        std::cerr << "  max_tokens defaults to 64\n";
        return 1;
    }
    std::string gguf_path = argv[1];
    std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    int max_new_tokens = (argc >= 4) ? std::stoi(argv[3]) : 64;

    // 1. Parse GGUF file
    std::cerr << "[1/7] Parsing GGUF file...\n";
    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    // 2. Extract config and build tokenizer
    std::cerr << "[2/7] Building tokenizer...\n";
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

    // Encode the prompt using chat format
    std::string chat_prompt =
        build_chat_prompt("You are a helpful assistant.", prompt, false);
    auto encoded = tokenizer.encode(chat_prompt, false, false);
    if (!encoded) {
        std::cerr << "Failed to encode prompt: " << encoded.error().to_string() << "\n";
        return 1;
    }

    std::cerr << "Prompt: " << prompt << "\n";
    std::cerr << "Encoded to " << encoded->size() << " tokens\n";
    std::cerr << "Config: layers=" << cfg.num_layers << " hidden=" << cfg.hidden_size
              << " heads=" << cfg.num_heads << " kv_heads=" << cfg.num_kv_heads
              << " head_dim=" << cfg.head_dim << " vocab=" << cfg.vocab_size << "\n";

    int prompt_len = static_cast<int>(encoded->size());
    int max_seq_len = prompt_len + max_new_tokens;

    // 3. Build prefill graph (seq_len = prompt_len)
    std::cerr << "[3/7] Building prefill graph (seq_len=" << prompt_len << ")...\n";
    TransformerConfig prefill_cfg = cfg;
    prefill_cfg.seq_len = prompt_len;

    Graph prefill_graph;
    GraphBuilder prefill_gb(prefill_graph);
    TransformerGraphBuilder prefill_tb(prefill_gb);

    auto prefill_result = prefill_tb.build_transformer(prefill_cfg);
    if (!prefill_result) {
        std::cerr << "Failed to build prefill graph: " << prefill_result.error().to_string() << "\n";
        return 1;
    }

    auto st = prefill_graph.validate();
    if (!st.ok()) {
        std::cerr << "Prefill graph validation failed: " << st.to_string() << "\n";
        return 1;
    }

    // 4. Build decode graph (seq_len = 1)
    std::cerr << "[4/7] Building decode graph (seq_len=1)...\n";
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

    st = decode_graph.validate();
    if (!st.ok()) {
        std::cerr << "Decode graph validation failed: " << st.to_string() << "\n";
        return 1;
    }

    // 5. Allocate runtime tensors and load shared weights once
    std::cerr << "[5/7] Allocating tensors and loading shared weights...\n";
    auto shared_weights = loader.load_shared_weights(*file_result, prefill_graph);
    if (!shared_weights) {
        std::cerr << "Failed to load shared weights: "
                  << shared_weights.error().to_string() << "\n";
        return 1;
    }
    std::cerr << "Shared weights: " << shared_weights->tensor_count()
              << " tensors, " << (shared_weights->total_bytes() / (1024 * 1024))
              << " MiB\n";

    // Prefill context
    RuntimeContext prefill_ctx;
    auto prefill_plan = allocate_runtime_tensors_cpu(prefill_graph, prefill_ctx);
    if (!prefill_plan) {
        std::cerr << "Failed to allocate prefill tensors: "
                  << prefill_plan.error().to_string() << "\n";
        return 1;
    }
    print_plan_summary("Prefill", *prefill_plan);
    st = shared_weights->bind(prefill_graph, prefill_ctx);
    if (!st.ok()) {
        std::cerr << "Failed to bind prefill weights: " << st.to_string() << "\n";
        return 1;
    }

    // Decode context
    RuntimeContext decode_ctx;
    auto decode_plan = allocate_runtime_tensors_cpu(decode_graph, decode_ctx);
    if (!decode_plan) {
        std::cerr << "Failed to allocate decode tensors: "
                  << decode_plan.error().to_string() << "\n";
        return 1;
    }
    print_plan_summary("Decode", *decode_plan);
    st = shared_weights->bind(decode_graph, decode_ctx);
    if (!st.ok()) {
        std::cerr << "Failed to bind decode weights: " << st.to_string() << "\n";
        return 1;
    }

    // 6. Initialize shared KV cache
    auto kv_cache = std::make_shared<KVCache>();
    kv_cache->init(static_cast<int>(cfg.num_layers),
                   static_cast<int>(cfg.num_kv_heads),
                   static_cast<int>(cfg.head_dim),
                   max_seq_len);

    prefill_ctx.set_kv_cache(kv_cache);
    decode_ctx.set_kv_cache(kv_cache);
    prefill_ctx.set_kv_cache_advance_tokens(prompt_len);
    decode_ctx.set_kv_cache_advance_tokens(1);

    // 7. Compile executors
    KernelRegistry registry;
    register_cpu_kernels(registry);
    auto backend = std::make_shared<CpuBackend>();

    CpuExecutor prefill_exec(backend, registry);
    st = prefill_exec.compile(prefill_graph);
    if (!st.ok()) {
        std::cerr << "Prefill compile failed: " << st.to_string() << "\n";
        return 1;
    }

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

    // ===== PREFILL =====
    std::cerr << "[6/7] Prefilling (" << prompt_len << " tokens)...\n";

    // Fill input_ids with prompt tokens
    for (const auto& v : prefill_graph.values()) {
        if (v.name == "input_ids" && v.kind == ValueKind::Input) {
            auto* t = prefill_ctx.get(v.id);
            auto* ptr = reinterpret_cast<int32_t*>(t->data());
            for (int64_t i = 0; i < prompt_len; ++i) {
                ptr[i] = (*encoded)[i];
            }
        }
    }

    // Run prefill
    st = prefill_exec.run(prefill_ctx);
    if (!st.ok()) {
        std::cerr << "Prefill failed: " << st.to_string() << "\n";
        return 1;
    }

    // Get logits at last position → sample first token
    auto* logits_tensor = prefill_ctx.get(*prefill_result);
    if (!logits_tensor) {
        std::cerr << "Output tensor not found\n";
        return 1;
    }

    auto* logits_data = reinterpret_cast<const float*>(logits_tensor->data());
    int64_t vocab_size = cfg.vocab_size;
    const float* last_logits = logits_data + (prompt_len - 1) * vocab_size;

    std::vector<int32_t> all_tokens = *encoded;
    int32_t first_token = sampler.sample(
        last_logits, static_cast<int32_t>(vocab_size),
        samp_cfg, all_tokens);

    std::cerr << "  prefill -> token " << first_token << "\n";

    if (first_token == tokenizer.eos_token_id()) {
        std::cerr << "Model produced EOS immediately after prefill\n";
        return 0;
    }
    all_tokens.push_back(first_token);

    // Decode and print first token
    const int32_t think_start_id = 151667;
    const int32_t think_end_id = 151668;
    bool in_thinking = false;

    if (first_token == think_start_id) {
        in_thinking = true;
        std::cerr << "[thinking] " << std::flush;
    } else if (first_token == think_end_id) {
        in_thinking = false;
        std::cerr << "\n" << std::flush;
    } else {
        auto decoded = tokenizer.decode({first_token});
        if (decoded) std::cout << *decoded << std::flush;
    }

    // ===== DECODE LOOP =====
    std::cerr << "[7/7] Decoding (" << (max_new_tokens - 1) << " more tokens)...\n\n";

    for (int step = 1; step < max_new_tokens; ++step) {
        // Fill decode input_ids with the last generated token
        for (const auto& v : decode_graph.values()) {
            if (v.name == "input_ids" && v.kind == ValueKind::Input) {
                auto* t = decode_ctx.get(v.id);
                auto* ptr = reinterpret_cast<int32_t*>(t->data());
                ptr[0] = all_tokens.back();
            }
        }

        // Run decode step
        st = decode_exec.run(decode_ctx);
        if (!st.ok()) {
            std::cerr << "Decode failed at step " << step << ": "
                      << st.to_string() << "\n";
            break;
        }

        // Get logits
        auto* dec_logits = decode_ctx.get(*decode_result);
        if (!dec_logits) {
            std::cerr << "Decode output tensor not found\n";
            break;
        }

        auto* dec_logits_data = reinterpret_cast<const float*>(dec_logits->data());

        int32_t next_token = sampler.sample(
            dec_logits_data, static_cast<int32_t>(vocab_size),
            samp_cfg, all_tokens);

        if (next_token == tokenizer.eos_token_id()) break;

        all_tokens.push_back(next_token);

        if (next_token == think_start_id) {
            in_thinking = true;
            std::cerr << "[thinking] " << std::flush;
            continue;
        }
        if (next_token == think_end_id) {
            in_thinking = false;
            std::cerr << "\n" << std::flush;
            continue;
        }
        if (in_thinking) continue;

        auto decoded = tokenizer.decode({next_token});
        if (decoded) std::cout << *decoded << std::flush;
    }

    std::cout << "\n\n[Generated " << (all_tokens.size() - encoded->size()) << " tokens total]\n";
    return 0;
}
