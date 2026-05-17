#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "minillm/core/tensor.h"
#include "minillm/graph/value.h"

namespace minillm {

class Graph;

// Binds ValueIds to physical Tensors at runtime.
// Owns the Tensors it allocates for intermediates; externally-owned
// Tensors (inputs, weights) are just referenced.
class RuntimeContext {
public:
    // Bind an externally-owned Tensor to a ValueId.
    Status bind(ValueId id, Tensor* tensor);

    // Bind and take ownership of a Tensor.
    Status emplace(ValueId id, std::unique_ptr<Tensor> tensor);

    // Get the Tensor bound to a ValueId, or nullptr.
    Tensor* get(ValueId id) const;

    // Allocate intermediate/output Tensors for all non-Input, non-Constant
    // Values in the graph that don't already have a binding.
    // Uses static shape; dynamic dims must already be resolved.
    Status allocate_intermediates(const Graph& graph);

private:
    // ValueId.value -> Tensor*
    std::unordered_map<size_t, Tensor*> bindings_;
    // Owning storage for tensors we allocate
    std::vector<std::unique_ptr<Tensor>> owned_;
};

} // namespace minillm
