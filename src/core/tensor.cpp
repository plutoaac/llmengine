#include "minillm/core/tensor.h"

namespace minillm {

Tensor::Tensor(std::string name, Shape shape, DType dtype, Device device)
    : name_(std::move(name)), shape_(std::move(shape)), dtype_(dtype), device_(device) {}

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

bool Tensor::is_allocated() const { return !storage_.empty(); }

void* Tensor::data() { return storage_.data(); }
const void* Tensor::data() const { return storage_.data(); }

Status Tensor::allocate_cpu() {
    if (shape_.has_dynamic_dim()) {
        return Status::invalid_argument(
            "cannot allocate tensor with dynamic shape: " + shape_.to_string());
    }
    auto nb = nbytes();
    if (!nb) return nb.error();
    storage_.resize(*nb);
    return Status::make_ok();
}

Status Tensor::allocate_cpu_bytes(size_t bytes) {
    storage_.resize(bytes);
    return Status::make_ok();
}

} // namespace minillm
