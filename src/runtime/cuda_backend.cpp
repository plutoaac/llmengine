#include "minillm/runtime/cuda_backend.h"

namespace minillm {

DeviceType CudaBackend::device_type() const { return DeviceType::CUDA; }

bool CudaBackend::supports(OpType op) const {
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

std::string_view CudaBackend::name() const { return "CudaBackend"; }

} // namespace minillm
