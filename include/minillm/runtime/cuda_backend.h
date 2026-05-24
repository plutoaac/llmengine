#pragma once

#include "minillm/runtime/backend.h"

namespace minillm {

class CudaBackend final : public Backend {
public:
    CudaBackend() : Backend(DeviceType::CUDA, "CudaBackend") {}
};

} // namespace minillm
