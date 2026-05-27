#pragma once

#include "minillm/runtime/kernel_registry.h"

namespace minillm {

// Registers all CPU kernel implementations into the KernelRegistry.
// Each kernel extracts input/output Tensor* from RuntimeContext,
// casts to float*, and delegates to cpu::* functions.
void register_cpu_kernels(KernelRegistry& registry);

} // namespace minillm
