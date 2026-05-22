#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"

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

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [seq_len]\n";
        return 1;
    }

    std::string gguf_path = argv[1];
    int64_t seq_len = (argc >= 3) ? std::stoll(argv[2]) : 4;
    if (seq_len <= 0) {
        std::cerr << "seq_len must be positive\n";
        return 1;
    }

    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice failed")) return 1;

    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    std::cout << "GGUF file parsed: version=" << file_result->version
              << " tensors=" << file_result->tensor_count
              << " metadata_keys=" << file_result->metadata.size() << "\n";

    auto cfg_result = loader.extract_config(*file_result);
    if (!cfg_result) {
        std::cerr << "Failed to extract config: " << cfg_result.error().to_string() << "\n";
        return 1;
    }

    auto cfg = *cfg_result;
    cfg.seq_len = seq_len;
    cfg.device = Device::cuda(0);
    cfg.dtype = DType::Float32;

    std::cout << "Config: batch=" << cfg.batch_size << " seq=" << cfg.seq_len
              << " vocab=" << cfg.vocab_size << " hidden=" << cfg.hidden_size
              << " inter=" << cfg.intermediate_size << " layers=" << cfg.num_layers
              << " heads=" << cfg.num_heads << " kv_heads=" << cfg.num_kv_heads
              << " head_dim=" << cfg.head_dim << " qk_norm=" << cfg.use_qk_norm
              << " eps=" << cfg.rms_norm_eps << "\n\n";

    Graph graph;
    GraphBuilder gb(graph);
    TransformerGraphBuilder tb(gb);

    auto result = tb.build_transformer(cfg);
    if (!result) {
        std::cerr << "Failed to build graph: " << result.error().to_string() << "\n";
        return 1;
    }

    auto st = graph.validate();
    if (!check_status(st, "Graph validation failed")) return 1;
    std::cout << "Graph built and validated (" << graph.values().size()
              << " values, " << graph.nodes().size() << " nodes)\n\n";

    RuntimeContext ctx;
    for (const auto& v : graph.values()) {
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        if (v.device.type == DeviceType::CUDA) {
            st = t->allocate_cuda();
        } else {
            st = t->allocate_cpu();
        }
        if (!check_status(st, "Failed to allocate " + v.name)) return 1;
        ctx.emplace(v.id, std::move(t));
    }
    std::cout << "All tensors allocated on CUDA\n\n";

    st = loader.load_weights(*file_result, graph, ctx);
    if (!check_status(st, "Failed to load weights")) return 1;
    std::cout << "Weights loaded from GGUF to CUDA tensors\n\n";

    std::vector<int32_t> input_ids(static_cast<size_t>(cfg.seq_len));
    for (int64_t i = 0; i < cfg.seq_len; ++i) {
        input_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i + 1);
    }

    for (const auto& v : graph.values()) {
        if (v.name == "input_ids" && v.kind == ValueKind::Input) {
            auto* t = ctx.get(v.id);
            if (!t) {
                std::cerr << "input_ids tensor not found\n";
                return 1;
            }
            if (!check_cuda(cudaMemcpy(t->data(), input_ids.data(),
                                       input_ids.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice),
                            "Failed to copy input_ids to CUDA")) {
                return 1;
            }
        }
    }

    KernelRegistry registry;
    register_cuda_kernels(registry);

    CudaExecutor exec(std::make_shared<CudaBackend>(), registry);
    st = exec.compile(graph);
    if (!check_status(st, "CUDA compile failed")) return 1;

    st = exec.run(ctx);
    if (!check_status(st, "CUDA forward pass failed")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "CUDA synchronize failed")) return 1;

    auto* logits_tensor = ctx.get(*result);
    if (!logits_tensor) {
        std::cerr << "Output tensor not found\n";
        return 1;
    }

    auto n = logits_tensor->numel();
    if (!n) {
        std::cerr << "Output numel failed: " << n.error().to_string() << "\n";
        return 1;
    }

    std::vector<float> logits(*n);
    if (!check_cuda(cudaMemcpy(logits.data(), logits_tensor->data(),
                               logits.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "Failed to copy logits to host")) {
        return 1;
    }

    int nan_count = 0;
    int inf_count = 0;
    float max_val = 0.0f;
    for (float v : logits) {
        if (std::isnan(v)) {
            ++nan_count;
        } else if (std::isinf(v)) {
            ++inf_count;
        } else {
            max_val = std::max(max_val, std::abs(v));
        }
    }

    std::cout << "CUDA forward pass completed!\n";
    std::cout << "Output shape: " << logits_tensor->shape().to_string() << "\n";
    std::cout << "Output dtype: " << dtype_name(logits_tensor->dtype()) << "\n";
    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "WARNING: output contains " << nan_count << " NaN, "
                  << inf_count << " Inf values\n";
    } else {
        std::cout << "Output max abs value: " << max_val << "\n";
        std::cout << "No NaN or Inf detected in output\n";
    }

    std::cout << "First 8 logits: ";
    for (size_t i = 0; i < std::min(logits.size(), size_t(8)); ++i) {
        std::cout << logits[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
