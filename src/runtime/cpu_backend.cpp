#include "minillm/runtime/cpu_backend.h"

namespace minillm {

DeviceType CpuBackend::device_type() const { return DeviceType::CPU; }

bool CpuBackend::supports(OpType op) const {
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

std::string_view CpuBackend::name() const { return "CpuBackend"; }

} // namespace minillm
