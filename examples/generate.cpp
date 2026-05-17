#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/io/bpe_tokenizer.h"
#include "minillm/runtime/sampler.h"

using namespace minillm;

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
    std::cerr << "[1/6] Parsing GGUF file...\n";
    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    // 2. Extract config and build tokenizer
    std::cerr << "[2/6] Building tokenizer...\n";
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
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n";
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

    // 3. Build graph
    // Since we don't have KV-cache, we use a multi-step approach:
    // Each step, run the full forward pass with only the real tokens (no padding).
    // This is slow but correct because causal attention only sees prior tokens.
    int64_t max_seq_len = static_cast<int64_t>(encoded->size()) + max_new_tokens;
    cfg.seq_len = max_seq_len;

    std::cerr << "[3/6] Building computation graph (seq_len=" << cfg.seq_len << ")...\n";
    Graph graph;
    GraphBuilder gb(graph);
    TransformerGraphBuilder tb(gb);

    auto result = tb.build_transformer(cfg);
    if (!result) {
        std::cerr << "Failed to build graph: " << result.error().to_string() << "\n";
        return 1;
    }

    auto st = graph.validate();
    if (!st.ok()) {
        std::cerr << "Graph validation failed: " << st.to_string() << "\n";
        return 1;
    }

    // 4. Allocate tensors and load weights
    std::cerr << "[4/6] Allocating tensors...\n";
    RuntimeContext ctx;
    for (const auto& v : graph.values()) {
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        st = t->allocate_cpu();
        if (!st.ok()) {
            std::cerr << "Failed to allocate " << v.name << ": " << st.to_string() << "\n";
            return 1;
        }
        ctx.emplace(v.id, std::move(t));
    }

    std::cerr << "[5/6] Loading weights...\n";
    st = loader.load_weights(*file_result, graph, ctx);
    if (!st.ok()) {
        std::cerr << "Failed to load weights: " << st.to_string() << "\n";
        return 1;
    }

    // 5. Autoregressive generation
    // Key insight: causal attention at position k only depends on tokens 0..k-1.
    // So a single forward pass gives correct logits for ALL positions.
    // We run ONE forward pass with the prompt, then iteratively:
    //   - Read the logit at the last real position
    //   - Append the sampled token
    //   - Re-run the full forward pass with the extended sequence
    // This is O(n^2) but correct without KV-cache.
    std::cerr << "[6/6] Generating (" << max_new_tokens << " tokens)...\n\n";

    KernelRegistry registry;
    register_cpu_kernels(registry);
    auto backend = std::make_shared<CpuBackend>();
    CpuExecutor exec(backend, registry);
    st = exec.compile(graph);
    if (!st.ok()) {
        std::cerr << "Compile failed: " << st.to_string() << "\n";
        return 1;
    }

    SamplingConfig samp_cfg;
    samp_cfg.greedy = true;
    samp_cfg.temperature = 0.8f;
    samp_cfg.top_k = 40;
    samp_cfg.top_p = 0.9f;
    samp_cfg.repetition_penalty = 1.1f;
    Sampler sampler(42);

    // Build the full token sequence (prompt + generated)
    std::vector<int32_t> all_tokens = *encoded;

    // Qwen3 thinking tokens
    const int32_t think_start_id = 151667;
    const int32_t think_end_id = 151668;
    bool in_thinking = false;

    for (int step = 0; step < max_new_tokens; ++step) {
        // Fill input_ids with all_tokens, pad rest with last token (repeat)
        // Using last token instead of EOS avoids attention corruption
        for (const auto& v : graph.values()) {
            if (v.name == "input_ids" && v.kind == ValueKind::Input) {
                auto* t = ctx.get(v.id);
                auto* ptr = reinterpret_cast<int32_t*>(t->data());
                for (int64_t i = 0; i < cfg.seq_len; ++i) {
                    if (i < (int64_t)all_tokens.size()) {
                        ptr[i] = all_tokens[i];
                    } else {
                        // Pad with the EOS token — since we only read logits at
                        // position (all_tokens.size()-1), the padding positions
                        // don't affect our output (causal mask ensures this)
                        ptr[i] = tokenizer.eos_token_id();
                    }
                }
            }
        }

        // Run forward pass
        st = exec.run(ctx);
        if (!st.ok()) {
            std::cerr << "Forward pass failed at step " << step << ": "
                      << st.to_string() << "\n";
            break;
        }

        // Get logits at the last real position
        auto* logits_tensor = ctx.get(*result);
        if (!logits_tensor) {
            std::cerr << "Output tensor not found\n";
            break;
        }

        int64_t last_pos = (int64_t)all_tokens.size() - 1;
        auto* logits_data = reinterpret_cast<const float*>(logits_tensor->data());
        int64_t vocab_size = cfg.vocab_size;
        const float* last_logits = logits_data + last_pos * vocab_size;

        // Sample next token
        int32_t next_token = sampler.sample(
            last_logits, static_cast<int32_t>(vocab_size),
            samp_cfg, all_tokens);

        // Check for EOS
        if (next_token == tokenizer.eos_token_id()) {
            break;
        }

        all_tokens.push_back(next_token);

        // Handle Qwen3 thinking mode
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

        // Decode and print non-thinking tokens
        auto decoded = tokenizer.decode({next_token});
        if (decoded) {
            std::cout << *decoded << std::flush;
        }
    }

    std::cout << "\n\n[Generated " << (all_tokens.size() - encoded->size()) << " tokens total]\n";
    return 0;
}
