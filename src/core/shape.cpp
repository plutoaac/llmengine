#include "minillm/core/shape.h"

namespace minillm {

Shape::Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

size_t Shape::rank() const { return dims_.size(); }

int64_t Shape::dim(size_t i) const { return dims_.at(i); }

const std::vector<int64_t>& Shape::dims() const { return dims_; }

bool Shape::empty() const { return dims_.empty(); }

bool Shape::has_dynamic_dim() const {
    for (auto d : dims_) {
        if (d < 0) return true;
    }
    return false;
}

std::expected<size_t, Status> Shape::numel() const {
    if (has_dynamic_dim()) {
        return std::unexpected(Status::invalid_argument(
            "cannot compute numel of shape with dynamic dims: " + to_string()));
    }
    size_t n = 1;
    for (auto d : dims_) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

std::string Shape::to_string() const {
    if (dims_.empty()) return "[]";
    std::string s = "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(dims_[i]);
    }
    s += "]";
    return s;
}

} // namespace minillm
