#pragma once

#include <string_view>

namespace minillm {

enum class OpType {
    Input,
    Constant,
    Embedding,
    MatMul,
    Linear,
    Add,
    Mul,
    RMSNorm,
    LayerNorm,
    SiLU,
    SwiGLU,
    RoPE,
    Attention,
    QKNorm,
    Softmax,
    Reshape,
    Transpose,
    View,
    Contiguous,
    Output,
    Custom,
};

std::string_view op_type_name(OpType type);

} // namespace minillm
