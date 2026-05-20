#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "minillm/core/tensor.h"
#include "minillm/graph/value.h"
#include "minillm/runtime/kv_cache.h"

namespace minillm {

class Graph;

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

    // KV cache access
    KVCache* kv_cache() const { return kv_cache_.get(); }
    void set_kv_cache(std::shared_ptr<KVCache> cache) { kv_cache_ = std::move(cache); }
    std::shared_ptr<KVCache> shared_kv_cache() { return kv_cache_; }
    void set_kv_cache_advance_tokens(int tokens) { kv_cache_advance_tokens_ = tokens; }
    int kv_cache_advance_tokens() const { return kv_cache_advance_tokens_; }
    Status advance_kv_cache_step();

private:
    // ValueId.value -> Tensor*
    std::unordered_map<size_t, Tensor*> bindings_;
    // Owning storage for tensors we allocate
    std::vector<std::unique_ptr<Tensor>> owned_;
    // KV cache (shared between prefill and decode contexts)
    std::shared_ptr<KVCache> kv_cache_;
    int kv_cache_advance_tokens_ = 0;
};

} // namespace minillm
