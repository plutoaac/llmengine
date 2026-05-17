#pragma once

#include <string_view>

#include "minillm/core/device.h"
#include "minillm/graph/op_type.h"

namespace minillm {

class Backend {
public:
    virtual ~Backend() = default;
    virtual DeviceType device_type() const = 0;
    virtual bool supports(OpType op) const = 0;
    virtual std::string_view name() const = 0;
};

} // namespace minillm
