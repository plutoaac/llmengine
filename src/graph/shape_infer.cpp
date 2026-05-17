#include "minillm/graph/shape_infer.h"

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

} // namespace minillm
