#pragma once

#include <string>
#include <variant>
#include <vector>

#include "minillm/core/shape.h"

namespace minillm {

using AttributeValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    Shape,
    std::vector<int64_t>>;

std::string attribute_to_string(const AttributeValue& value);

} // namespace minillm
