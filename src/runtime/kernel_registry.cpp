#include "minillm/runtime/kernel_registry.h"

namespace minillm {

void KernelRegistry::register_kernel(DeviceType device, OpType op, KernelFn fn) {
    kernels_[{device, op}] = std::move(fn);
}

std::expected<KernelFn, Status> KernelRegistry::find(DeviceType device, OpType op) const {
    auto it = kernels_.find({device, op});
    if (it != kernels_.end()) return it->second;
    return std::unexpected(Status::not_implemented(
        "no kernel registered for op " + std::string(op_type_name(op))));
}

bool KernelRegistry::has(DeviceType device, OpType op) const {
    return kernels_.find({device, op}) != kernels_.end();
}

} // namespace minillm
