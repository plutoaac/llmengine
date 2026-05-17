#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "minillm/core/status.h"
#include "minillm/graph/graph.h"
#include "minillm/io/gguf_parser.h"
#include "minillm/model/transformer_graph_builder.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

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
