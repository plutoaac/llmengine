#include "minillm/core/shape.h"
#include <climits>
#include <limits>
#include <stdexcept>

namespace minillm {

Shape::Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
    for (auto d : dims_) {
        if (d < 0 && d != -1) {
            throw std::invalid_argument(
                "Shape dimension must be >= 0 or -1 (dynamic), got: " + std::to_string(d));
        }
        if (d == 0) {
            throw std::invalid_argument("Shape dimension must not be zero");
        }
    }
}

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
        if (d > 0 && n > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
            return std::unexpected(Status::invalid_argument(
                "numel overflow for shape: " + to_string()));
        }
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
