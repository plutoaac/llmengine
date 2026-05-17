#include "minillm/runtime/runtime_context.h"

#include "minillm/graph/graph.h"

namespace minillm {

Status RuntimeContext::bind(ValueId id, Tensor* tensor) {
    if (!tensor) {
        return Status::invalid_argument("cannot bind null Tensor to ValueId %" +
                                        std::to_string(id.value));
    }
    bindings_[id.value] = tensor;
    return Status::make_ok();
}

Status RuntimeContext::emplace(ValueId id, std::unique_ptr<Tensor> tensor) {
    if (!tensor) {
        return Status::invalid_argument("cannot emplace null Tensor for ValueId %" +
                                        std::to_string(id.value));
    }
    bindings_[id.value] = tensor.get();
    owned_.push_back(std::move(tensor));
    return Status::make_ok();
}

Tensor* RuntimeContext::get(ValueId id) const {
    auto it = bindings_.find(id.value);
    if (it != bindings_.end()) return it->second;
    return nullptr;
}

Status RuntimeContext::allocate_intermediates(const Graph& graph) {
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Input || v.kind == ValueKind::Constant) continue;
        if (bindings_.count(v.id.value)) continue;
        if (v.shape.has_dynamic_dim()) {
            return Status::invalid_argument(
                "cannot allocate intermediate with dynamic shape: " + v.name +
                " " + v.shape.to_string());
        }
        auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
        auto st = t->allocate_cpu();
        if (!st.ok()) return st;
        bindings_[v.id.value] = t.get();
        owned_.push_back(std::move(t));
    }
    return Status::make_ok();
}

} // namespace minillm
