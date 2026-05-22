#include "minillm/runtime/runtime_context.h"

#include <cstdint>
#include <limits>
#include <unordered_map>

#include "minillm/graph/graph.h"

namespace minillm {

namespace {

std::byte* align_pointer(std::byte* ptr, size_t alignment) {
    if (alignment == 0) return ptr;
    auto raw = reinterpret_cast<std::uintptr_t>(ptr);
    const auto rem = raw % alignment;
    if (rem == 0) return ptr;
    return reinterpret_cast<std::byte*>(raw + (alignment - rem));
}

Status allocate_owned_tensor(const Value& v, std::unique_ptr<Tensor>& out) {
    if (v.shape.has_dynamic_dim()) {
        return Status::invalid_argument(
            "cannot allocate intermediate with dynamic shape: " + v.name +
            " " + v.shape.to_string());
    }
    auto t = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
    Status st;
    if (v.device.type == DeviceType::CUDA) {
        st = t->allocate_cuda();
    } else {
        st = t->allocate_cpu();
    }
    if (!st.ok()) return st;
    out = std::move(t);
    return Status::make_ok();
}

} // namespace

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
        std::unique_ptr<Tensor> t;
        auto st = allocate_owned_tensor(v, t);
        if (!st.ok()) return st;
        bindings_[v.id.value] = t.get();
        owned_.push_back(std::move(t));
    }
    return Status::make_ok();
}

std::expected<MemoryPlan, Status> RuntimeContext::allocate_intermediates_planned(
    const Graph& graph,
    MemoryPlanOptions options) {
    auto plan = MemoryPlanner::plan(graph, options);
    if (!plan) return std::unexpected(plan.error());

    auto st = allocate_intermediates_with_plan(graph, *plan);
    if (!st.ok()) return std::unexpected(st);

    return *plan;
}

Status RuntimeContext::allocate_intermediates_with_plan(
    const Graph& graph,
    const MemoryPlan& plan) {
    constexpr size_t kNoBuffer = static_cast<size_t>(-1);
    const size_t alignment = plan.alignment == 0 ? size_t{64} : plan.alignment;

    std::vector<std::byte*> buffer_data(plan.buffers.size(), nullptr);
    std::vector<size_t> buffer_bytes(plan.buffers.size(), 0);

    for (const auto& buffer : plan.buffers) {
        if (buffer.id >= plan.buffers.size()) {
            return Status::out_of_range("memory plan buffer id is out of range");
        }
        if (buffer.bytes > std::numeric_limits<size_t>::max() - alignment) {
            return Status::out_of_range("CPU arena allocation size overflow");
        }

        CpuArenaBlock block;
        block.bytes = buffer.bytes;
        block.storage = std::make_unique<std::byte[]>(buffer.bytes + alignment);
        block.data = align_pointer(block.storage.get(), alignment);
        buffer_data[buffer.id] = block.data;
        buffer_bytes[buffer.id] = buffer.bytes;
        cpu_arenas_.push_back(std::move(block));
    }

    for (const auto& range : plan.ranges) {
        if (!range.eligible || range.buffer_id == kNoBuffer) continue;
        if (range.buffer_id >= buffer_data.size() || !buffer_data[range.buffer_id]) {
            return Status::out_of_range(
                "memory plan range references missing buffer for " + range.name);
        }
        if (bindings_.count(range.value.value)) continue;

        auto value = graph.value(range.value);
        if (!value) return value.error();
        auto tensor = std::make_unique<Tensor>(
            (*value)->name, (*value)->shape, (*value)->dtype, (*value)->device);
        auto st = tensor->bind_cpu_data(
            buffer_data[range.buffer_id], buffer_bytes[range.buffer_id]);
        if (!st.ok()) return st;
        bindings_[range.value.value] = tensor.get();
        owned_.push_back(std::move(tensor));
    }

    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Input || v.kind == ValueKind::Constant) continue;
        if (bindings_.count(v.id.value)) continue;
        std::unique_ptr<Tensor> t;
        auto st = allocate_owned_tensor(v, t);
        if (!st.ok()) return st;
        bindings_[v.id.value] = t.get();
        owned_.push_back(std::move(t));
    }

    return Status::make_ok();
}

Status RuntimeContext::advance_kv_cache_step() {
    if (kv_cache_advance_tokens_ <= 0) {
        return Status::make_ok();
    }
    if (!kv_cache_ || !kv_cache_->initialized()) {
        return Status::runtime_error("cannot advance an uninitialized KV cache");
    }
    if (!kv_cache_->can_append(kv_cache_advance_tokens_)) {
        return Status::out_of_range("KV cache does not have enough space to advance");
    }

    kv_cache_->advance(kv_cache_advance_tokens_);
    return Status::make_ok();
}

} // namespace minillm
