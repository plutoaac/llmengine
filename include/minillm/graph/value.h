#pragma once

#include <string>

#include "minillm/core/device.h"
#include "minillm/core/dtype.h"
#include "minillm/core/shape.h"

namespace minillm {

struct ValueId {
    size_t value{0};
    bool operator==(const ValueId& o) const { return value == o.value; }
};

struct NodeId {
    size_t value{0};
    bool operator==(const NodeId& o) const { return value == o.value; }
};

enum class ValueKind {
    Input,
    Constant,
    Intermediate,
    Output,
};

struct Value {
    ValueId id;
    std::string name;
    Shape shape;
    DType dtype{DType::Unknown};
    Device device{Device::cpu()};
    ValueKind kind{ValueKind::Intermediate};
};

} // namespace minillm
