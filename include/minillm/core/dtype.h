#pragma once

#include <cstddef>
#include <expected>
#include <string_view>

#include "minillm/core/status.h"

namespace minillm {

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt8,
    Int8,
    Bool,
    Unknown,
};

std::string_view dtype_name(DType type);
std::expected<size_t, Status> dtype_size(DType type);
bool is_floating_point(DType type);
bool is_integer(DType type);

} // namespace minillm
