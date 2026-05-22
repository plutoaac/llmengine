#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <unordered_map>

#include "minillm/core/status.h"
#include "minillm/core/tensor.h"
#include "minillm/graph/graph.h"
#include "minillm/io/gguf_parser.h"
#include "minillm/model/transformer_graph_builder.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

class SharedWeightStore {
public:
    SharedWeightStore() = default;
    SharedWeightStore(const SharedWeightStore&) = delete;
    SharedWeightStore& operator=(const SharedWeightStore&) = delete;
    SharedWeightStore(SharedWeightStore&&) noexcept = default;
    SharedWeightStore& operator=(SharedWeightStore&&) noexcept = default;

    // Bind graph Constant values by name to already-loaded shared tensors.
    Status bind(const Graph& graph, RuntimeContext& ctx) const;

    Tensor* get(const std::string& value_name) const;
    size_t tensor_count() const { return storage_by_gguf_name_.size(); }
    size_t alias_count() const { return aliases_by_value_name_.size(); }
    size_t total_bytes() const { return total_bytes_; }

private:
    friend class WeightLoader;

    std::unordered_map<std::string, std::unique_ptr<Tensor>> storage_by_gguf_name_;
    std::unordered_map<std::string, Tensor*> aliases_by_value_name_;
    size_t total_bytes_ = 0;
};

class WeightLoader {
public:
    explicit WeightLoader(std::string gguf_path);

    // Parse the GGUF file header. Must be called first.
    std::expected<GGUFFile, Status> open();

    // Extract TransformerConfig from GGUF metadata.
    std::expected<TransformerConfig, Status> extract_config(
        const GGUFFile& file) const;

    // Load all weight tensors from the GGUF file into the RuntimeContext.
    // Matches GGUF tensor names to Graph Constant values via name mapping.
    // Dequantizes F16/BF16 to F32 if the target Tensor is Float32.
    Status load_weights(
        const GGUFFile& file,
        const Graph& graph,
        RuntimeContext& ctx) const;

    // Load each GGUF tensor once and bind it into multiple RuntimeContexts.
    // This is intended for prefill/decode graphs that share the same weights.
    std::expected<SharedWeightStore, Status> load_shared_weights(
        const GGUFFile& file,
        const Graph& graph) const;

    // Map GGUF tensor name to MiniLLMEngine Value name.
    static std::expected<std::string, Status> gguf_name_to_value_name(
        const std::string& gguf_name, int layer_index = 0);

    // Dequantize F16/BF16 source data to F32 destination.
    static Status dequantize_to_f32(
        GgmlDataType src_dtype,
        const void* src,
        float* dst,
        size_t num_elements);

private:
    std::string gguf_path_;
};

} // namespace minillm
