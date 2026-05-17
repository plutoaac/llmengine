#include <cmath>
#include <iostream>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"

using namespace minillm;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }
    std::string gguf_path = argv[1];

    // 1. Parse GGUF file
    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    std::cout << "GGUF file parsed: version=" << file_result->version
              << " tensors=" << file_result->tensor_count
              << " metadata_keys=" << file_result->metadata.size() << "\n";

    // 2. Extract TransformerConfig from metadata
    auto cfg_result = loader.extract_config(*file_result);
    if (!cfg_result) {
        std::cerr << "Failed to extract config: " << cfg_result.error().to_string() << "\n";
        return 1;
    }
    auto cfg = *cfg_result;
    cfg.seq_len = 4; // override for small test

    std::cout << "Config: batch=" << cfg.batch_size << " seq=" << cfg.seq_len
              << " vocab=" << cfg.vocab_size << " hidden=" << cfg.hidden_size
              << " inter=" << cfg.intermediate_size << " layers=" << cfg.num_layers
              << " heads=" << cfg.num_heads << " kv_heads=" << cfg.num_kv_heads
              << " head_dim=" << cfg.head_dim << " qk_norm=" << cfg.use_qk_norm
              << " eps=" << cfg.rms_norm_eps << "\n\n";

    // 3. Build graph
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
    std::cout << "Graph built and validated (" << graph.values().size()
              << " values, " << graph.nodes().size() << " nodes)\n\n";

    // 4. Allocate tensors in RuntimeContext
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
    std::cout << "All tensors allocated\n\n";

    // 5. Load weights from GGUF
    st = loader.load_weights(*file_result, graph, ctx);
    if (!st.ok()) {
        std::cerr << "Failed to load weights: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << "Weights loaded from GGUF\n\n";

    // 6. Fill input_ids with test token IDs
    for (const auto& v : graph.values()) {
        if (v.name == "input_ids" && v.kind == ValueKind::Input) {
            auto* t = ctx.get(v.id);
            auto* ptr = reinterpret_cast<int32_t*>(t->data());
            for (int64_t i = 0; i < cfg.seq_len; ++i) {
                ptr[i] = static_cast<int32_t>(i + 1);
            }
        }
    }

    // 7. Compile and run
    KernelRegistry registry;
    register_cpu_kernels(registry);

    auto backend = std::make_shared<CpuBackend>();
    CpuExecutor exec(backend, registry);

    st = exec.compile(graph);
    if (!st.ok()) {
        std::cerr << "Compile failed: " << st.to_string() << "\n";
        return 1;
    }

    st = exec.run(ctx);
    if (!st.ok()) {
        std::cerr << "Forward pass failed: " << st.to_string() << "\n";
        return 1;
    }

    // 8. Verify output
    auto* logits_tensor = ctx.get(*result);
    if (!logits_tensor) {
        std::cerr << "Output tensor not found\n";
        return 1;
    }

    std::cout << "Forward pass completed!\n";
    std::cout << "Output shape: " << logits_tensor->shape().to_string() << "\n";
    std::cout << "Output dtype: " << dtype_name(logits_tensor->dtype()) << "\n";

    auto n = logits_tensor->numel();
    if (n) {
        const auto* ptr = reinterpret_cast<const float*>(logits_tensor->data());
        int nan_count = 0, inf_count = 0;
        float max_val = 0.0f;
        for (size_t i = 0; i < *n; ++i) {
            if (std::isnan(ptr[i])) ++nan_count;
            else if (std::isinf(ptr[i])) ++inf_count;
            else max_val = std::max(max_val, std::abs(ptr[i]));
        }
        if (nan_count > 0 || inf_count > 0) {
            std::cerr << "WARNING: output contains " << nan_count << " NaN, "
                      << inf_count << " Inf values\n";
        } else {
            std::cout << "Output max abs value: " << max_val << "\n";
            std::cout << "No NaN or Inf detected in output\n";
        }

        // Print first few logits
        std::cout << "First 8 logits: ";
        for (size_t i = 0; i < std::min(*n, size_t(8)); ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
