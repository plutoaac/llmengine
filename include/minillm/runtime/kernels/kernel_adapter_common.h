#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <variant>

#include "minillm/core/dtype.h"
#include "minillm/core/status.h"
#include "minillm/core/tensor.h"
#include "minillm/graph/node.h"
#include "minillm/runtime/runtime_context.h"
#include "minillm/utils/bfloat16.hpp"

namespace minillm {
namespace detail {

// --- Data accessors ---

inline const float* float_data(const Tensor* t) {
    return reinterpret_cast<const float*>(t->data());
}

inline float* float_data_mut(Tensor* t) {
    return reinterpret_cast<float*>(t->data());
}

inline const int* int_data(const Tensor* t) {
    return reinterpret_cast<const int*>(t->data());
}

// --- Validation helpers ---

inline Status check_allocated(const Tensor* t, std::string_view name) {
    if (!t->is_allocated())
        return Status::runtime_error(std::string(name) + " tensor not allocated");
    return Status::make_ok();
}

inline Status check_dtype_float(const Tensor* t, std::string_view name) {
    if (t->dtype() != DType::Float32)
        return Status::unsupported(
            std::string(name) + " only supports Float32, got " +
            std::string(dtype_name(t->dtype())));
    return Status::make_ok();
}

inline Status check_dtype_floating(const Tensor* t, std::string_view name) {
    if (!is_floating_point(t->dtype()))
        return Status::unsupported(
            std::string(name) + " must be a floating tensor, got " +
            std::string(dtype_name(t->dtype())));
    return Status::make_ok();
}

inline const bfloat16_t* bf16_data(const Tensor* t) {
    return reinterpret_cast<const bfloat16_t*>(t->data());
}

inline const uint16_t* bf16_bits_data(const Tensor* t) {
    return reinterpret_cast<const uint16_t*>(t->data());
}

inline const uint8_t* q8_data(const Tensor* t) {
    return reinterpret_cast<const uint8_t*>(t->data());
}

inline std::expected<Tensor*, Status> get_tensor(ValueId id, RuntimeContext& ctx,
                                                   std::string_view role) {
    auto* t = ctx.get(id);
    if (!t)
        return std::unexpected(Status::runtime_error(
            std::string(role) + " tensor not found for ValueId %" +
            std::to_string(id.value)));
    return t;
}

// --- Dimension/size helpers ---

inline std::expected<int, Status> checked_int(size_t value, std::string_view role) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        return std::unexpected(Status::unsupported(
            std::string(role) + " is too large for int indexing"));
    return static_cast<int>(value);
}

inline std::expected<int, Status> checked_dim(int64_t value, std::string_view role) {
    if (value < 0 || value > std::numeric_limits<int>::max())
        return std::unexpected(Status::unsupported(
            std::string(role) + " dimension is outside int range"));
    return static_cast<int>(value);
}

inline std::expected<int, Status> numel_int(const Tensor* t, std::string_view role) {
    auto n = t->numel();
    if (!n) return std::unexpected(n.error());
    return checked_int(*n, role);
}

inline std::expected<int, Status> rows_before_last_dim(const Tensor* t, std::string_view role) {
    if (t->shape().rank() == 0)
        return std::unexpected(Status::shape_mismatch(
            std::string(role) + " expects rank >= 1"));
    size_t rows = 1;
    for (size_t i = 0; i + 1 < t->shape().rank(); ++i)
        rows *= static_cast<size_t>(t->shape().dim(i));
    return checked_int(rows, role);
}

inline std::expected<int, Status> last_dim(const Tensor* t, std::string_view role) {
    if (t->shape().rank() == 0)
        return std::unexpected(Status::shape_mismatch(
            std::string(role) + " expects rank >= 1"));
    return checked_dim(t->shape().dim(t->shape().rank() - 1), role);
}

// --- Attribute extraction ---

// Concept: types that can be stored in AttributeValue variant.
template<typename T>
concept AttrType = std::same_as<T, int64_t> || std::same_as<T, double>
                || std::same_as<T, bool> || std::same_as<T, std::string>;

// Extract a typed attribute from a Node, returning fallback if missing or wrong type.
template<AttrType T>
inline T attr(const Node& node, std::string_view name, T fallback) {
    if (auto a = node.get_attr(name))
        if (auto* p = std::get_if<T>(&*a)) return *p;
    return fallback;
}

} // namespace detail
} // namespace minillm
