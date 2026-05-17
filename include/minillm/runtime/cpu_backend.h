#pragma once

#include "minillm/runtime/backend.h"

namespace minillm {

class CpuBackend final : public Backend {
public:
    DeviceType device_type() const override;
    bool supports(OpType op) const override;
    std::string_view name() const override;
};

} // namespace minillm
