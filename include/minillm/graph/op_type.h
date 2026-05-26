#pragma once

#include <string_view>
#include <utility>

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

constexpr std::string_view op_type_name(OpType type) {
    switch (type) {
    case OpType::Input:      return "Input";
    case OpType::Constant:   return "Constant";
    case OpType::Embedding:  return "Embedding";
    case OpType::MatMul:     return "MatMul";
    case OpType::Linear:     return "Linear";
    case OpType::Add:        return "Add";
    case OpType::Mul:        return "Mul";
    case OpType::RMSNorm:    return "RMSNorm";
    case OpType::LayerNorm:  return "LayerNorm";
    case OpType::SiLU:       return "SiLU";
    case OpType::SwiGLU:     return "SwiGLU";
    case OpType::RoPE:       return "RoPE";
    case OpType::Attention:  return "Attention";
    case OpType::QKNorm:     return "QKNorm";
    case OpType::Softmax:    return "Softmax";
    case OpType::Reshape:    return "Reshape";
    case OpType::Transpose:  return "Transpose";
    case OpType::View:       return "View";
    case OpType::Contiguous: return "Contiguous";
    case OpType::Output:     return "Output";
    case OpType::Custom:     return "Custom";
    default:                 std::unreachable();
    }
}

} // namespace minillm
