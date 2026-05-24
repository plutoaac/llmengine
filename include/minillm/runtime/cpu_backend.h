#pragma once

#include "minillm/runtime/backend.h"

namespace minillm {

class CpuBackend final : public Backend {
public:
    CpuBackend() : Backend(DeviceType::CPU, "CpuBackend") {}
};

} // namespace minillm
