#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "minillm/core/tensor.h"
#include "minillm/graph/value.h"
#include "minillm/runtime/kv_cache.h"
#include "minillm/runtime/memory_planner.h"
#include "minillm/runtime/paged_kv_cache.h"

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

    // Allocate non-input/non-constant values with CPU intermediate buffer reuse.
    // Returns the plan used for arena bindings so callers can report memory savings.
    std::expected<MemoryPlan, Status> allocate_intermediates_planned(
        const Graph& graph,
        MemoryPlanOptions options = {});

    // Apply an existing memory plan to this context. Values not covered by the
    // plan, such as outputs by default, fall back to individual allocation.
    Status allocate_intermediates_with_plan(
        const Graph& graph,
        const MemoryPlan& plan);

    // KV cache access
    KVCache* kv_cache() const { return kv_cache_.get(); }
    void set_kv_cache(std::shared_ptr<KVCache> cache) { kv_cache_ = std::move(cache); }
    std::shared_ptr<KVCache> shared_kv_cache() { return kv_cache_; }
    void set_kv_cache_advance_tokens(int tokens) { kv_cache_advance_tokens_ = tokens; }
    int kv_cache_advance_tokens() const { return kv_cache_advance_tokens_; }
    Status advance_kv_cache_step();

    // Paged KV cache for paged attention decode.
    // When set, the attention kernel reads K/V from the paged cache and
    // writes newly projected K/V to it, bypassing the contiguous cache.
    PagedKVCache* paged_kv_cache() const { return paged_kv_cache_; }
    void set_paged_kv_cache(PagedKVCache* cache) { paged_kv_cache_ = cache; }
    int paged_sequence_id() const { return paged_sequence_id_; }
    void set_paged_sequence_id(int id) { paged_sequence_id_ = id; }
    int paged_kv_write_pos() const { return paged_kv_write_pos_; }
    void set_paged_kv_write_pos(int pos) { paged_kv_write_pos_ = pos; }
    void reset_paged_kv_write_pos() { paged_kv_write_pos_ = -1; }

private:
    struct CpuArenaBlock {
        std::unique_ptr<std::byte[]> storage;
        std::byte* data = nullptr;
        size_t bytes = 0;
    };

    // ValueId.value -> Tensor*
    std::unordered_map<size_t, Tensor*> bindings_;
    // Backing storage for planned CPU intermediate reuse.
    std::vector<CpuArenaBlock> cpu_arenas_;
    // Owning storage for tensors we allocate
    std::vector<std::unique_ptr<Tensor>> owned_;
    // KV cache (shared between prefill and decode contexts)
    std::shared_ptr<KVCache> kv_cache_;
    int kv_cache_advance_tokens_ = 0;

    // Paged KV cache (non-owning; lifetime managed by caller)
    PagedKVCache* paged_kv_cache_ = nullptr;
    int paged_sequence_id_ = 0;
    int paged_kv_write_pos_ = -1;
};

} // namespace minillm
