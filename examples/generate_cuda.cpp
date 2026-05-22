#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/bpe_tokenizer.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/runtime/sampler.h"

using namespace minillm;

namespace {

bool check_status(const Status& st, const std::string& what) {
    if (st.ok()) return true;
    std::cerr << what << ": " << st.to_string() << "\n";
    return false;
}

bool check_cuda(cudaError_t err, const std::string& what) {
    if (err == cudaSuccess) return true;
    std::cerr << what << ": " << cudaGetErrorString(err) << "\n";
    return false;
}

Status allocate_runtime_tensors_cuda(const Graph& graph, RuntimeContext& ctx) {
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Constant) continue;
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        Status st;
        if (v.device.type == DeviceType::CUDA) {
            st = t->allocate_cuda();
        } else {
            st = t->allocate_cpu();
        }
        if (!st.ok()) return st;
        st = ctx.emplace(v.id, std::move(t));
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status copy_input_ids_to_cuda(const Graph& graph, RuntimeContext& ctx,
                              const std::vector<int32_t>& ids) {
    for (const auto& v : graph.values()) {
        if (v.name != "input_ids" || v.kind != ValueKind::Input) continue;
        auto* t = ctx.get(v.id);
        if (!t) {
            return Status::runtime_error("input_ids tensor not found");
        }
        auto err = cudaMemcpy(t->data(), ids.data(), ids.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return Status::runtime_error(
                "copy input_ids to CUDA failed: " +
                std::string(cudaGetErrorString(err)));
        }
        return Status::make_ok();
    }
    return Status::not_found("input_ids value not found in graph");
}

std::expected<std::vector<float>, Status> copy_logits_to_host(
    const Tensor* logits, size_t offset, size_t count) {
    if (!logits) {
        return std::unexpected(Status::runtime_error("logits tensor not found"));
    }
    std::vector<float> host(count);
    const auto* src = reinterpret_cast<const float*>(logits->data()) + offset;
    auto err = cudaMemcpy(host.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return std::unexpected(Status::runtime_error(
            "copy CUDA logits to host failed: " + std::string(cudaGetErrorString(err))));
    }
    return host;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [prompt] [max_tokens]\n";
        return 1;
    }

    std::string gguf_path = argv[1];
    std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    int max_new_tokens = (argc >= 4) ? std::stoi(argv[3]) : 16;
    if (max_new_tokens <= 0) {
        std::cerr << "max_tokens must be positive\n";
        return 1;
    }

    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice failed")) return 1;

    std::cerr << "[1/7] Parsing GGUF file...\n";
    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    std::cerr << "[2/7] Building tokenizer...\n";
    auto cfg_result = loader.extract_config(*file_result);
    if (!cfg_result) {
        std::cerr << "Failed to extract config: " << cfg_result.error().to_string() << "\n";
        return 1;
    }
    auto cfg = *cfg_result;
    cfg.device = Device::cuda(0);
    cfg.dtype = DType::Float32;

    BPETokenizer tokenizer;
    auto tok_st = tokenizer.init(*file_result);
    if (!tok_st) {
        std::cerr << "Failed to init tokenizer: " << tok_st.error().to_string() << "\n";
        return 1;
    }

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

    const int prompt_len = static_cast<int>(encoded->size());
    const int max_seq_len = prompt_len + max_new_tokens;

    std::cerr << "[3/7] Building CUDA prefill graph (seq_len=" << prompt_len << ")...\n";
    TransformerConfig prefill_cfg = cfg;
    prefill_cfg.seq_len = prompt_len;
    Graph prefill_graph;
    GraphBuilder prefill_gb(prefill_graph);
    TransformerGraphBuilder prefill_tb(prefill_gb);
    auto prefill_result = prefill_tb.build_transformer(prefill_cfg);
    if (!prefill_result) {
        std::cerr << "Failed to build prefill graph: "
                  << prefill_result.error().to_string() << "\n";
        return 1;
    }
    auto st = prefill_graph.validate();
    if (!check_status(st, "Prefill graph validation failed")) return 1;

    std::cerr << "[4/7] Building CUDA decode graph (seq_len=1)...\n";
    TransformerConfig decode_cfg = cfg;
    decode_cfg.seq_len = 1;
    Graph decode_graph;
    GraphBuilder decode_gb(decode_graph);
    TransformerGraphBuilder decode_tb(decode_gb);
    auto decode_result = decode_tb.build_transformer(decode_cfg);
    if (!decode_result) {
        std::cerr << "Failed to build decode graph: "
                  << decode_result.error().to_string() << "\n";
        return 1;
    }
    st = decode_graph.validate();
    if (!check_status(st, "Decode graph validation failed")) return 1;

    std::cerr << "[5/7] Allocating CUDA tensors and loading shared weights...\n";
    auto shared_weights = loader.load_shared_weights(*file_result, prefill_graph);
    if (!shared_weights) {
        std::cerr << "Failed to load shared weights: "
                  << shared_weights.error().to_string() << "\n";
        return 1;
    }
    std::cerr << "Shared CUDA weights: " << shared_weights->tensor_count()
              << " tensors, " << (shared_weights->total_bytes() / (1024 * 1024))
              << " MiB\n";

    RuntimeContext prefill_ctx;
    st = allocate_runtime_tensors_cuda(prefill_graph, prefill_ctx);
    if (!check_status(st, "Failed to allocate prefill tensors")) return 1;
    st = shared_weights->bind(prefill_graph, prefill_ctx);
    if (!check_status(st, "Failed to bind prefill weights")) return 1;

    RuntimeContext decode_ctx;
    st = allocate_runtime_tensors_cuda(decode_graph, decode_ctx);
    if (!check_status(st, "Failed to allocate decode tensors")) return 1;
    st = shared_weights->bind(decode_graph, decode_ctx);
    if (!check_status(st, "Failed to bind decode weights")) return 1;

    auto kv_cache = std::make_shared<KVCache>();
    st = kv_cache->init_cuda(static_cast<int>(cfg.num_layers),
                             static_cast<int>(cfg.num_kv_heads),
                             static_cast<int>(cfg.head_dim),
                             max_seq_len, 0);
    if (!check_status(st, "Failed to initialize CUDA KV cache")) return 1;

    prefill_ctx.set_kv_cache(kv_cache);
    decode_ctx.set_kv_cache(kv_cache);
    prefill_ctx.set_kv_cache_advance_tokens(prompt_len);
    decode_ctx.set_kv_cache_advance_tokens(1);

    KernelRegistry registry;
    register_cuda_kernels(registry);
    auto backend = std::make_shared<CudaBackend>();

    CudaExecutor prefill_exec(backend, registry);
    st = prefill_exec.compile(prefill_graph);
    if (!check_status(st, "Prefill CUDA compile failed")) return 1;
    CudaExecutor decode_exec(backend, registry);
    st = decode_exec.compile(decode_graph);
    if (!check_status(st, "Decode CUDA compile failed")) return 1;

    SamplingConfig samp_cfg;
    samp_cfg.greedy = true;
    samp_cfg.temperature = 0.8f;
    samp_cfg.top_k = 40;
    samp_cfg.top_p = 0.9f;
    samp_cfg.repetition_penalty = 1.1f;
    Sampler sampler(42);

    std::cerr << "[6/7] CUDA prefilling (" << prompt_len << " tokens)...\n";
    st = copy_input_ids_to_cuda(prefill_graph, prefill_ctx, *encoded);
    if (!check_status(st, "Failed to copy prefill input_ids")) return 1;
    st = prefill_exec.run(prefill_ctx);
    if (!check_status(st, "CUDA prefill failed")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "CUDA prefill synchronize failed")) return 1;

    const int64_t vocab_size = cfg.vocab_size;
    auto prefill_logits = copy_logits_to_host(
        prefill_ctx.get(*prefill_result),
        static_cast<size_t>(prompt_len - 1) * static_cast<size_t>(vocab_size),
        static_cast<size_t>(vocab_size));
    if (!prefill_logits) {
        std::cerr << prefill_logits.error().to_string() << "\n";
        return 1;
    }

    std::vector<int32_t> all_tokens = *encoded;
    int32_t next_token = sampler.sample(
        prefill_logits->data(), static_cast<int32_t>(vocab_size),
        samp_cfg, all_tokens);
    if (next_token == tokenizer.eos_token_id()) {
        std::cerr << "Model produced EOS immediately after prefill\n";
        return 0;
    }
    all_tokens.push_back(next_token);

    const int32_t think_start_id = 151667;
    const int32_t think_end_id = 151668;
    bool in_thinking = false;
    if (next_token == think_start_id) {
        in_thinking = true;
        std::cerr << "[thinking] " << std::flush;
    } else if (next_token != think_end_id) {
        auto decoded = tokenizer.decode({next_token});
        if (decoded) std::cout << *decoded << std::flush;
    }

    std::cerr << "[7/7] CUDA decoding (" << (max_new_tokens - 1)
              << " more tokens)...\n\n";

    for (int step = 1; step < max_new_tokens; ++step) {
        std::vector<int32_t> decode_ids{all_tokens.back()};
        st = copy_input_ids_to_cuda(decode_graph, decode_ctx, decode_ids);
        if (!check_status(st, "Failed to copy decode input_ids")) return 1;
        st = decode_exec.run(decode_ctx);
        if (!check_status(st, "CUDA decode failed")) return 1;
        if (!check_cuda(cudaDeviceSynchronize(), "CUDA decode synchronize failed")) return 1;

        auto decode_logits = copy_logits_to_host(
            decode_ctx.get(*decode_result), 0, static_cast<size_t>(vocab_size));
        if (!decode_logits) {
            std::cerr << decode_logits.error().to_string() << "\n";
            return 1;
        }

        next_token = sampler.sample(
            decode_logits->data(), static_cast<int32_t>(vocab_size),
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

    std::cout << "\n\n[Generated " << (all_tokens.size() - encoded->size())
              << " tokens total on CUDA]\n";
    return 0;
}
