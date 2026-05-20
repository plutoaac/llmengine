#pragma once

#include "minillm/runtime/kernel_registry.h"

namespace minillm {

// Registers CUDA kernels into the KernelRegistry. CUDA support is optional and
// only built when MINILLM_ENABLE_CUDA is enabled in CMake.
void register_cuda_kernels(KernelRegistry& registry);

} // namespace minillm
