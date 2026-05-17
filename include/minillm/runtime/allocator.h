#pragma once

#include <cstddef>
#include <expected>

#include "minillm/core/device.h"
#include "minillm/core/status.h"

namespace minillm {

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual std::expected<void*, Status> allocate(size_t bytes, Device device) = 0;
    virtual void deallocate(void* ptr, Device device) = 0;
};

class CpuAllocator final : public Allocator {
public:
    std::expected<void*, Status> allocate(size_t bytes, Device device) override;
    void deallocate(void* ptr, Device device) override;
};

} // namespace minillm
