#pragma once

#include <cstdint>
#include <expected>

#include "minillm/core/dtype.h"
#include "minillm/core/status.h"
#include "minillm/graph/value.h"

namespace minillm {

class GraphBuilder;

struct TransformerConfig {
    int64_t batch_size = 1;
    int64_t seq_len = -1;
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t intermediate_size = 2048;
    int64_t num_layers = 1;
    int64_t num_heads = 12;
    int64_t num_kv_heads = 12;
    int64_t head_dim = 64;
    double rms_norm_eps = 1e-6;
    bool use_qk_norm = false;
    DType dtype = DType::Float32;
};

class TransformerGraphBuilder {
public:
    explicit TransformerGraphBuilder(GraphBuilder& builder);

    std::expected<ValueId, Status> build_tiny_decoder_block(
        const TransformerConfig& config);

    std::expected<ValueId, Status> build_transformer(
        const TransformerConfig& config);

private:
    GraphBuilder& builder_;
};

} // namespace minillm
