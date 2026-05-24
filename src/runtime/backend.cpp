#include "minillm/runtime/backend.h"

namespace minillm {

Backend::Backend(DeviceType dt, std::string_view n) : device_type_(dt), name_(n) {}

DeviceType Backend::device_type() const { return device_type_; }

bool Backend::supports(OpType op) const {
    switch (op) {
    case OpType::Embedding:
    case OpType::MatMul:
    case OpType::Linear:
    case OpType::Add:
    case OpType::Mul:
    case OpType::RMSNorm:
    case OpType::SiLU:
    case OpType::SwiGLU:
    case OpType::RoPE:
    case OpType::Attention:
    case OpType::QKNorm:
    case OpType::Softmax:
    case OpType::Reshape:
    case OpType::Transpose:
    case OpType::Output:
        return true;
    default:
        return false;
    }
}

std::string_view Backend::name() const { return name_; }

} // namespace minillm
