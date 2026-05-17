#pragma once

#include <expected>

#include "minillm/core/shape.h"
#include "minillm/core/status.h"

namespace minillm {

std::expected<Shape, Status> infer_matmul_shape(const Shape& a, const Shape& b);
std::expected<Shape, Status> infer_linear_shape(const Shape& x, const Shape& weight);
std::expected<Shape, Status> infer_add_shape(const Shape& a, const Shape& b);
std::expected<Shape, Status> infer_same_shape_unary(const Shape& x);
std::expected<Shape, Status> infer_embedding_shape(const Shape& input_ids, const Shape& weight);

} // namespace minillm
