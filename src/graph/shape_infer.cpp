#include "minillm/graph/shape_infer.h"

#include <algorithm>

namespace minillm {

std::expected<Shape, Status> infer_matmul_shape(const Shape& a, const Shape& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        return std::unexpected(Status::shape_mismatch(
            "MatMul expects rank-2 inputs, got " + a.to_string() +
            " and " + b.to_string()));
    }
    auto k_a = a.dim(1);
    auto k_b = b.dim(0);
    // Allow dynamic dim to pass through; only reject static mismatch
    if (k_a >= 0 && k_b >= 0 && k_a != k_b) {
        return std::unexpected(Status::shape_mismatch(
            "MatMul inner dims mismatch: " + a.to_string() +
            " x " + b.to_string()));
    }
    return Shape({a.dim(0), b.dim(1)});
}

std::expected<Shape, Status> infer_linear_shape(const Shape& x, const Shape& weight) {
    if (weight.rank() != 2) {
        return std::unexpected(Status::shape_mismatch(
            "Linear weight must be rank 2, got " + weight.to_string()));
    }
    if (x.rank() < 1) {
        return std::unexpected(Status::shape_mismatch(
            "Linear input must have rank >= 1, got " + x.to_string()));
    }
    auto in_feat = x.dim(x.rank() - 1);
    auto w_in = weight.dim(1);
    if (in_feat >= 0 && w_in >= 0 && in_feat != w_in) {
        return std::unexpected(Status::shape_mismatch(
            "Linear in_features mismatch: x " + x.to_string() +
            " vs weight " + weight.to_string()));
    }
    std::vector<int64_t> out_dims;
    for (size_t i = 0; i + 1 < x.rank(); ++i) {
        out_dims.push_back(x.dim(i));
    }
    out_dims.push_back(weight.dim(0));
    return Shape(std::move(out_dims));
}

std::expected<Shape, Status> infer_add_shape(const Shape& a, const Shape& b) {
    if (a.rank() != b.rank()) {
        return std::unexpected(Status::shape_mismatch(
            "Add requires same rank: " + a.to_string() +
            " vs " + b.to_string()));
    }
    for (size_t i = 0; i < a.rank(); ++i) {
        auto da = a.dim(i), db = b.dim(i);
        if (da >= 0 && db >= 0 && da != db) {
            return std::unexpected(Status::shape_mismatch(
                "Add dim mismatch at axis " + std::to_string(i) + ": " +
                a.to_string() + " vs " + b.to_string()));
        }
    }
    return a;
}

std::expected<Shape, Status> infer_same_shape_unary(const Shape& x) {
    return x;
}

std::expected<Shape, Status> infer_embedding_shape(const Shape& input_ids, const Shape& weight) {
    if (input_ids.rank() < 1 || weight.rank() != 2) {
        return std::unexpected(Status::shape_mismatch(
            "Embedding expects input_ids rank>=1 and weight rank 2, got " +
            input_ids.to_string() + " and " + weight.to_string()));
    }
    std::vector<int64_t> out_dims;
    for (size_t i = 0; i < input_ids.rank(); ++i) {
        out_dims.push_back(input_ids.dim(i));
    }
    out_dims.push_back(weight.dim(1));
    return Shape(std::move(out_dims));
}

std::expected<Shape, Status> infer_reshape_shape(const Shape& input, const Shape& target) {
    int infer_axis = -1;
    int64_t known_product = 1;
    for (size_t i = 0; i < target.rank(); ++i) {
        int64_t dim = target.dim(i);
        if (dim == -1) {
            if (infer_axis >= 0) {
                return std::unexpected(Status::shape_mismatch(
                    "Reshape target can contain at most one inferred dim: " +
                    target.to_string()));
            }
            infer_axis = static_cast<int>(i);
        } else if (dim <= 0) {
            return std::unexpected(Status::shape_mismatch(
                "Reshape target dims must be positive or -1: " +
                target.to_string()));
        } else {
            known_product *= dim;
        }
    }

    if (input.has_dynamic_dim()) {
        return target;
    }

    auto input_numel = input.numel();
    if (!input_numel) return std::unexpected(input_numel.error());

    std::vector<int64_t> out_dims = target.dims();
    if (infer_axis >= 0) {
        if (known_product == 0 || static_cast<int64_t>(*input_numel) % known_product != 0) {
            return std::unexpected(Status::shape_mismatch(
                "Reshape cannot infer target dim for " + input.to_string() +
                " -> " + target.to_string()));
        }
        out_dims[static_cast<size_t>(infer_axis)] =
            static_cast<int64_t>(*input_numel) / known_product;
        return Shape(std::move(out_dims));
    }

    auto target_numel = target.numel();
    if (!target_numel) return std::unexpected(target_numel.error());
    if (*input_numel != *target_numel) {
        return std::unexpected(Status::shape_mismatch(
            "Reshape element count mismatch: " + input.to_string() +
            " -> " + target.to_string()));
    }
    return target;
}

std::expected<Shape, Status> infer_transpose_shape(const Shape& input, int64_t axis0, int64_t axis1) {
    const int64_t rank = static_cast<int64_t>(input.rank());
    if (rank <= 0) {
        return std::unexpected(Status::shape_mismatch(
            "Transpose expects rank >= 1, got " + input.to_string()));
    }

    if (axis0 < 0) axis0 += rank;
    if (axis1 < 0) axis1 += rank;
    if (axis0 < 0 || axis0 >= rank || axis1 < 0 || axis1 >= rank) {
        return std::unexpected(Status::out_of_range(
            "Transpose axes out of range for shape " + input.to_string()));
    }

    std::vector<int64_t> out_dims = input.dims();
    std::swap(out_dims[static_cast<size_t>(axis0)], out_dims[static_cast<size_t>(axis1)]);
    return Shape(std::move(out_dims));
}

} // namespace minillm
