#include "minillm/runtime/allocator.h"

#include <cstdlib>

namespace minillm {

std::expected<void*, Status> CpuAllocator::allocate(size_t bytes, Device device) {
    if (device.type != DeviceType::CPU) {
        return std::unexpected(Status::unsupported(
            "CpuAllocator only supports CPU device, got " + device.to_string()));
    }
    if (bytes == 0) return nullptr;
    // Round up to multiple of 64 for aligned_alloc
    size_t aligned_bytes = ((bytes + 63) / 64) * 64;
    void* ptr = std::aligned_alloc(64, aligned_bytes);
    if (!ptr) {
        return std::unexpected(Status::runtime_error(
            "CpuAllocator failed to allocate " + std::to_string(bytes) + " bytes"));
    }
    return ptr;
}

void CpuAllocator::deallocate(void* ptr, Device device) {
    (void)device;
    std::free(ptr);
}

} // namespace minillm
