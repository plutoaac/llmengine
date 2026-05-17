#pragma once

#include "minillm/core/shape.h"

namespace minillm {

// Lightweight non-owning view over typed tensor data.
// Intended for CPU kernel invocation; later can be replaced by std::mdspan.
template <typename T>
class TensorView {
public:
    TensorView(T* data, Shape shape) : data_(data), shape_(std::move(shape)) {}

    T* data() { return data_; }
    const T* data() const { return data_; }
    const Shape& shape() const { return shape_; }

private:
    T* data_{nullptr};
    Shape shape_;
};

} // namespace minillm
