#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "minillm/core/device.h"
#include "minillm/core/dtype.h"
#include "minillm/core/shape.h"
#include "minillm/core/status.h"

namespace minillm {

// Runtime tensor: owns CPU/CUDA storage or views externally-owned CPU storage.
// In the Graph IR, logical tensors are represented by Value; Tensor is the
// physical data container that an Executor binds to a ValueId.
class Tensor {
public:
    Tensor() = default;
    Tensor(std::string name, Shape shape, DType dtype, Device device = Device::cpu());
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const std::string& name() const;
    const Shape& shape() const;
    DType dtype() const;
    const Device& device() const;

    std::expected<size_t, Status> numel() const;
    std::expected<size_t, Status> nbytes() const;

    bool is_allocated() const;
    void* data();
    const void* data() const;

    Status allocate_cpu();
    Status allocate_cuda();
    Status release();
    Status allocate_cpu_bytes(size_t bytes);
    Status bind_cpu_data(void* data, size_t bytes);
    Status bind_cuda_data(void* data, size_t bytes);

private:
    std::string name_;
    Shape shape_;
    DType dtype_{DType::Unknown};
    Device device_{Device::cpu()};
    std::vector<std::byte> storage_;
    void* external_data_{nullptr};
    size_t external_bytes_{0};
    void* cuda_storage_{nullptr};
    size_t cuda_bytes_{0};
};

} // namespace minillm
