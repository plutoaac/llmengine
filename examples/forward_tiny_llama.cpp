#include <cmath>
#include <iostream>
#include <random>

#include "minillm/minillm.h"

using namespace minillm;

static void fill_random_float(Tensor& t, float lo, float hi, uint32_t seed) {
    auto n = t.numel();
    if (!n) return;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* ptr = reinterpret_cast<float*>(t.data());
    for (size_t i = 0; i < *n; ++i) {
        ptr[i] = dist(gen);
    }
}

static void fill_random_int(Tensor& t, int lo, int hi, uint32_t seed) {
    auto n = t.numel();
    if (!n) return;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    auto* ptr = reinterpret_cast<int*>(t.data());
    for (size_t i = 0; i < *n; ++i) {
        ptr[i] = dist(gen);
    }
}

int main() {
    // Small dimensions for fast forward pass
    const int batch = 1;
    const int seq_len = 4;
    const int vocab_size = 64;
    const int hidden = 32;
    const int intermediate = 48;
    const int num_heads = 4;
    const int head_dim = 8;

    std::cout << "=== MiniLLMEngine Forward Pass ===\n";
    std::cout << "Config: batch=" << batch << " seq=" << seq_len
              << " vocab=" << vocab_size << " hidden=" << hidden
              << " inter=" << intermediate << " heads=" << num_heads
              << " head_dim=" << head_dim << "\n\n";

    // 1. Build graph
    Graph graph;
    GraphBuilder gb(graph);

    TransformerConfig cfg;
    cfg.batch_size = batch;
    cfg.seq_len = seq_len;
    cfg.vocab_size = vocab_size;
    cfg.hidden_size = hidden;
    cfg.intermediate_size = intermediate;
    cfg.num_heads = num_heads;
    cfg.num_kv_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.rms_norm_eps = 1e-6;

    TransformerGraphBuilder tb(gb);
    auto result = tb.build_tiny_decoder_block(cfg);
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

    // 2. Create RuntimeContext, allocate and fill all tensors
    RuntimeContext ctx;

    for (const auto& v : graph.values()) {
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        st = t->allocate_cpu();
        if (!st.ok()) {
            std::cerr << "Failed to allocate " << v.name << ": " << st.to_string() << "\n";
            return 1;
        }

        if (v.dtype == DType::Float32) {
            fill_random_float(*t, -0.5f, 0.5f,
                              static_cast<uint32_t>(v.id.value * 17 + 42));
        } else if (v.dtype == DType::Int32) {
            fill_random_int(*t, 0, vocab_size - 1,
                            static_cast<uint32_t>(v.id.value * 13 + 7));
        }

        ctx.emplace(v.id, std::move(t));
    }
    std::cout << "All tensors allocated and filled with random data\n\n";

    // 3. Register CPU kernels and compile
    KernelRegistry registry;
    register_cpu_kernels(registry);

    auto backend = std::make_shared<CpuBackend>();
    CpuExecutor exec(backend, registry);

    st = exec.compile(graph);
    if (!st.ok()) {
        std::cerr << "Compile failed: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << "CpuExecutor compiled OK\n\n";

    // 4. Run forward pass
    st = exec.run(ctx);
    if (!st.ok()) {
        std::cerr << "Forward pass failed: " << st.to_string() << "\n";
        return 1;
    }

    // 5. Verify output
    auto* logits_tensor = ctx.get(*result);
    if (!logits_tensor) {
        std::cerr << "Output tensor not found\n";
        return 1;
    }

    std::cout << "Forward pass completed!\n";
    std::cout << "Output shape: " << logits_tensor->shape().to_string() << "\n";
    std::cout << "Output dtype: " << dtype_name(logits_tensor->dtype()) << "\n";

    // Check for NaN/Inf
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
    }

    // Print first few logits
    if (n) {
        const auto* ptr = reinterpret_cast<const float*>(logits_tensor->data());
        std::cout << "First 8 logits: ";
        for (size_t i = 0; i < std::min(*n, size_t(8)); ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
