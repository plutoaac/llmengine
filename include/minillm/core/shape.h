#pragma once

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "minillm/core/status.h"

namespace minillm {

// Shape with -1 denoting a dynamic (runtime-resolved) dimension.
// E.g. [1, -1, 768] is typical for LLM batch-seq-hidden shapes.
class Shape {
public:
    Shape() = default;
    explicit Shape(std::vector<int64_t> dims);

    constexpr size_t rank() const { return dims_.size(); }
    int64_t dim(size_t i) const;
    const std::vector<int64_t>& dims() const;

    constexpr bool empty() const { return dims_.empty(); }
    bool has_dynamic_dim() const;
    std::expected<size_t, Status> numel() const;

    std::string to_string() const;

    bool operator==(const Shape& other) const = default;

private:
    std::vector<int64_t> dims_;
};

} // namespace minillm
