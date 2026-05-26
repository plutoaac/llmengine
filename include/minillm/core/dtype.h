#pragma once

#include <cstddef>
#include <expected>
#include <string_view>
#include <utility>

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

constexpr std::string_view dtype_name(DType type) {
    switch (type) {
    case DType::Float32:  return "Float32";
    case DType::Float16:  return "Float16";
    case DType::BFloat16: return "BFloat16";
    case DType::Int32:    return "Int32";
    case DType::Int64:    return "Int64";
    case DType::UInt8:    return "UInt8";
    case DType::Int8:     return "Int8";
    case DType::Bool:     return "Bool";
    case DType::Unknown:  return "Unknown";
    default:              std::unreachable();
    }
}

inline std::expected<size_t, Status> dtype_size(DType type) {
    switch (type) {
    case DType::Float32:  return 4;
    case DType::Float16:  return 2;
    case DType::BFloat16: return 2;
    case DType::Int32:    return 4;
    case DType::Int64:    return 8;
    case DType::UInt8:    return 1;
    case DType::Int8:     return 1;
    case DType::Bool:     return 1;
    case DType::Unknown:  return std::unexpected(Status::unsupported("Unknown dtype has no size"));
    default:              std::unreachable();
    }
}

constexpr bool is_floating_point(DType type) {
    return type == DType::Float32 || type == DType::Float16 || type == DType::BFloat16;
}

constexpr bool is_integer(DType type) {
    return type == DType::Int32 || type == DType::Int64 ||
           type == DType::UInt8 || type == DType::Int8;
}

} // namespace minillm
