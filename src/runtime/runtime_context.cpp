#include "minillm/runtime/runtime_context.h"

#include <cstdint>
#include <limits>
#include <unordered_map>

#include "minillm/graph/graph.h"

#if defined(MINILLM_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace minillm {

RuntimeContext::~RuntimeContext() {
#if defined(MINILLM_ENABLE_CUDA)
    for (auto& block : cuda_arenas_)
        if (block.data) cudaFree(block.data);
    cuda_arenas_.clear();
#endif
}

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
    TRY(st);
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
        TRY(allocate_owned_tensor(v, t));
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

    // Collect buffers by arena_index. All buffers in the same arena share one
    // contiguous heap allocation, reducing malloc overhead and fragmentation.
    std::unordered_map<size_t, std::vector<size_t>> arena_buffers;
    size_t max_arena = 0;
    for (size_t bi = 0; bi < plan.buffers.size(); ++bi) {
        const auto& buffer = plan.buffers[bi];
        if (buffer.arena_index == kNoBuffer) {
            return Status::runtime_error("buffer has no arena assignment: #" +
                                         std::to_string(buffer.id));
        }
        arena_buffers[buffer.arena_index].push_back(bi);
        max_arena = std::max(max_arena, buffer.arena_index);
    }

    // For each arena, compute total size = max(offset + bytes) across its buffers
    // and allocate one contiguous block.
    std::vector<std::byte*> arena_bases(max_arena + 1, nullptr);
    std::vector<size_t> arena_total_bytes(max_arena + 1, 0);
    // Track whether each arena is CUDA (index into cuda_arenas_) or CPU (index into cpu_arenas_)
    std::vector<DeviceType> arena_device(max_arena + 1, DeviceType::CPU);

    for (auto& [arena_idx, buffer_indices] : arena_buffers) {
        size_t total = 0;
        // Determine device from the first buffer in this arena
        const auto& first_buffer = plan.buffers[buffer_indices.front()];
        const DeviceType dt = first_buffer.device.type;
        arena_device[arena_idx] = dt;

        for (size_t bi : buffer_indices) {
            const auto& buffer = plan.buffers[bi];
            total = std::max(total, buffer.offset + buffer.bytes);
        }
        if (total > std::numeric_limits<size_t>::max() - alignment) {
            return Status::out_of_range("arena allocation size overflow");
        }

        if (dt == DeviceType::CUDA) {
#if defined(MINILLM_ENABLE_CUDA)
            CudaArenaBlock block;
            block.bytes = total;
            cudaError_t err = cudaSetDevice(first_buffer.device.index);
            if (err != cudaSuccess)
                return Status::runtime_error("cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
            err = cudaMalloc(&block.data, total + alignment);
            if (err != cudaSuccess)
                return Status::runtime_error("cudaMalloc arena failed: " + std::string(cudaGetErrorString(err)));
            std::byte* base = reinterpret_cast<std::byte*>(block.data);
            arena_bases[arena_idx] = align_pointer(base, alignment);
            arena_total_bytes[arena_idx] = total;
            cuda_arenas_.push_back(std::move(block));
#else
            return Status::unsupported("CUDA arena allocation requires CUDA support");
#endif
        } else {
            CpuArenaBlock block;
            block.bytes = total;
            block.storage = std::make_unique<std::byte[]>(total + alignment);
            block.data = align_pointer(block.storage.get(), alignment);
            arena_bases[arena_idx] = block.data;
            arena_total_bytes[arena_idx] = total;
            cpu_arenas_.push_back(std::move(block));
        }
    }

    // Map each buffer to its data pointer within the arena
    std::vector<std::byte*> buffer_data(plan.buffers.size(), nullptr);
    std::vector<size_t> buffer_avail_bytes(plan.buffers.size(), 0);
    for (size_t bi = 0; bi < plan.buffers.size(); ++bi) {
        const auto& buffer = plan.buffers[bi];
        buffer_data[bi] = arena_bases[buffer.arena_index] + buffer.offset;
        // avail_bytes for bind_cpu_data = bytes from this offset to arena end
        buffer_avail_bytes[bi] = arena_total_bytes[buffer.arena_index] - buffer.offset;
    }

    // Bind planned ranges to arena-backed tensors
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
        Status st;
        const auto& buffer = plan.buffers[range.buffer_id];
        if (arena_device[buffer.arena_index] == DeviceType::CUDA) {
            st = tensor->bind_cuda_data(
                buffer_data[range.buffer_id], buffer_avail_bytes[range.buffer_id]);
        } else {
            st = tensor->bind_cpu_data(
                buffer_data[range.buffer_id], buffer_avail_bytes[range.buffer_id]);
        }
        TRY(st);
        bindings_[range.value.value] = tensor.get();
        owned_.push_back(std::move(tensor));
    }

    // Fallback: allocate non-planned values individually
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::Input || v.kind == ValueKind::Constant) continue;
        if (bindings_.count(v.id.value)) continue;
        std::unique_ptr<Tensor> t;
        TRY(allocate_owned_tensor(v, t));
        bindings_[v.id.value] = t.get();
        owned_.push_back(std::move(t));
    }

    return Status::make_ok();
}

Status RuntimeContext::advance_kv_cache_step() {
    auto reset_paged_write_pos = [this]() {
        paged_kv_write_pos_ = -1;
    };

    if (kv_cache_advance_tokens_ <= 0) {
        reset_paged_write_pos();
        return Status::make_ok();
    }
    if (!kv_cache_ || !kv_cache_->initialized()) {
        reset_paged_write_pos();
        return Status::runtime_error("cannot advance an uninitialized KV cache");
    }
    if (!kv_cache_->can_append(kv_cache_advance_tokens_)) {
        reset_paged_write_pos();
        return Status::out_of_range("KV cache does not have enough space to advance");
    }

    kv_cache_->advance(kv_cache_advance_tokens_);
    reset_paged_write_pos();
    return Status::make_ok();
}

} // namespace minillm
