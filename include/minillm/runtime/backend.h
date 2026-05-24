#pragma once

#include <string_view>

#include "minillm/core/device.h"
#include "minillm/graph/op_type.h"

namespace minillm {

class Backend {
public:
    Backend(DeviceType dt, std::string_view n);
    virtual ~Backend() = default;
    DeviceType device_type() const;
    bool supports(OpType op) const;
    std::string_view name() const;

private:
    DeviceType device_type_;
    std::string_view name_;
};

} // namespace minillm
