#include "minillm/core/tensor.h"

#if defined(MINILLM_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace minillm {

Tensor::Tensor(std::string name, Shape shape, DType dtype, Device device)
    : name_(std::move(name)), shape_(std::move(shape)), dtype_(dtype), device_(device) {}

Tensor::~Tensor() {
    (void)release();
}

Tensor::Tensor(Tensor&& other) noexcept
    : name_(std::move(other.name_)),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_),
      storage_(std::move(other.storage_)),
      cuda_storage_(other.cuda_storage_),
      cuda_bytes_(other.cuda_bytes_) {
    other.cuda_storage_ = nullptr;
    other.cuda_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    (void)release();
    name_ = std::move(other.name_);
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    device_ = other.device_;
    storage_ = std::move(other.storage_);
    cuda_storage_ = other.cuda_storage_;
    cuda_bytes_ = other.cuda_bytes_;
    other.cuda_storage_ = nullptr;
    other.cuda_bytes_ = 0;
    return *this;
}

const std::string& Tensor::name() const { return name_; }
const Shape& Tensor::shape() const { return shape_; }
DType Tensor::dtype() const { return dtype_; }
const Device& Tensor::device() const { return device_; }

std::expected<size_t, Status> Tensor::numel() const { return shape_.numel(); }

std::expected<size_t, Status> Tensor::nbytes() const {
    auto n = shape_.numel();
    if (!n) return std::unexpected(n.error());
    auto s = dtype_size(dtype_);
    if (!s) return std::unexpected(s.error());
    return *n * *s;
}

bool Tensor::is_allocated() const { return !storage_.empty() || cuda_storage_ != nullptr; }

void* Tensor::data() {
    if (cuda_storage_) return cuda_storage_;
    return storage_.data();
}

const void* Tensor::data() const {
    if (cuda_storage_) return cuda_storage_;
    return storage_.data();
}

Status Tensor::allocate_cpu() {
    auto st = release();
    if (!st.ok()) return st;
    if (shape_.has_dynamic_dim()) {
        return Status::invalid_argument(
            "cannot allocate tensor with dynamic shape: " + shape_.to_string());
    }
    auto nb = nbytes();
    if (!nb) return nb.error();
    storage_.resize(*nb);
    device_ = Device::cpu();
    return Status::make_ok();
}

Status Tensor::allocate_cuda() {
    auto st = release();
    if (!st.ok()) return st;
    if (shape_.has_dynamic_dim()) {
        return Status::invalid_argument(
            "cannot allocate tensor with dynamic shape: " + shape_.to_string());
    }
    auto nb = nbytes();
    if (!nb) return nb.error();
#if defined(MINILLM_ENABLE_CUDA)
    cudaError_t err = cudaSetDevice(device_.index);
    if (err != cudaSuccess) {
        return Status::runtime_error(
            "cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&cuda_storage_, *nb);
    if (err != cudaSuccess) {
        cuda_storage_ = nullptr;
        cuda_bytes_ = 0;
        return Status::runtime_error(
            "cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    cuda_bytes_ = *nb;
    device_ = Device::cuda(device_.index);
    return Status::make_ok();
#else
    return Status::unsupported("MiniLLMEngine was built without CUDA support");
#endif
}

Status Tensor::release() {
#if defined(MINILLM_ENABLE_CUDA)
    if (cuda_storage_) {
        cudaError_t err = cudaFree(cuda_storage_);
        cuda_storage_ = nullptr;
        cuda_bytes_ = 0;
        if (err != cudaSuccess) {
            return Status::runtime_error(
                "cudaFree failed: " + std::string(cudaGetErrorString(err)));
        }
    }
#else
    cuda_storage_ = nullptr;
    cuda_bytes_ = 0;
#endif
    storage_.clear();
    storage_.shrink_to_fit();
    return Status::make_ok();
}

Status Tensor::allocate_cpu_bytes(size_t bytes) {
    auto st = release();
    if (!st.ok()) return st;
    storage_.resize(bytes);
    device_ = Device::cpu();
    return Status::make_ok();
}

} // namespace minillm
