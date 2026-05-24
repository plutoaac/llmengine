#include "minillm/runtime/executor.h"

#include "minillm/graph/graph.h"
#include "minillm/graph/op_type.h"
#include "minillm/runtime/backend.h"
#include "minillm/runtime/kernel_registry.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

// --- CpuExecutor ---

CpuExecutor::CpuExecutor(std::shared_ptr<Backend> backend, KernelRegistry& registry)
    : backend_(std::move(backend)), registry_(registry) {}

Status CpuExecutor::compile(const Graph& graph) {
    auto st = graph.validate();
    if (!st.ok()) return st;

    auto order = graph.topological_sort();
    if (!order) return order.error();

    for (const auto& nid : *order) {
        auto nd = graph.node(nid);
        if (!nd) return nd.error();

        auto op = (*nd)->op_type();
        if (op == OpType::Input || op == OpType::Constant || op == OpType::Output) continue;

        if (!backend_->supports(op)) {
            return Status::unsupported(
                "backend " + std::string(backend_->name()) +
                " does not support op " + std::string(op_type_name(op)));
        }
        if (!registry_.has(DeviceType::CPU, op)) {
            return Status::not_implemented(
                "no kernel registered for op " + std::string(op_type_name(op)));
        }
    }

    graph_ = &graph;
    execution_order_ = std::move(*order);
    return Status::make_ok();
}

Status CpuExecutor::run(RuntimeContext& ctx) {
    if (!graph_) {
        return Status::runtime_error("executor not compiled");
    }
    for (const auto& nid : execution_order_) {
        auto nd = graph_->node(nid);
        if (!nd) return nd.error();

        auto op = (*nd)->op_type();
        if (op == OpType::Input || op == OpType::Constant) continue;

        if (op == OpType::Output) {
            // Output is a no-op; the data is already in the input ValueId's Tensor.
            continue;
        }

        auto kernel = registry_.find(DeviceType::CPU, op);
        if (!kernel) return kernel.error();

        auto st = (*kernel)(**nd, ctx);
        if (!st.ok()) {
            return Status::runtime_error(
                "kernel " + std::string(op_type_name(op)) +
                " failed for node #" + std::to_string(nid.value) +
                " (" + std::string((*nd)->name()) + "): " + st.to_string());
        }
    }
    auto st = ctx.advance_kv_cache_step();
    if (!st.ok()) {
        return Status::runtime_error("KV cache advance failed: " + st.to_string());
    }
    return Status::make_ok();
}

#if defined(MINILLM_ENABLE_CUDA)

// --- CudaExecutor ---

CudaExecutor::CudaExecutor(std::shared_ptr<Backend> backend, KernelRegistry& registry)
    : backend_(std::move(backend)), registry_(registry) {}

Status CudaExecutor::compile(const Graph& graph) {
    auto st = graph.validate();
    if (!st.ok()) return st;

    auto order = graph.topological_sort();
    if (!order) return order.error();

    for (const auto& nid : *order) {
        auto nd = graph.node(nid);
        if (!nd) return nd.error();

        auto op = (*nd)->op_type();
        if (op == OpType::Input || op == OpType::Constant || op == OpType::Output) continue;

        if (!backend_->supports(op)) {
            return Status::unsupported(
                "backend " + std::string(backend_->name()) +
                " does not support op " + std::string(op_type_name(op)));
        }
        if (!registry_.has(DeviceType::CUDA, op)) {
            return Status::not_implemented(
                "no CUDA kernel registered for op " + std::string(op_type_name(op)));
        }
    }

    graph_ = &graph;
    execution_order_ = std::move(*order);
    return Status::make_ok();
}

Status CudaExecutor::run(RuntimeContext& ctx) {
    if (!graph_) {
        return Status::runtime_error("executor not compiled");
    }
    for (const auto& nid : execution_order_) {
        auto nd = graph_->node(nid);
        if (!nd) return nd.error();

        auto op = (*nd)->op_type();
        if (op == OpType::Input || op == OpType::Constant || op == OpType::Output) continue;

        auto kernel = registry_.find(DeviceType::CUDA, op);
        if (!kernel) return kernel.error();

        auto st = (*kernel)(**nd, ctx);
        if (!st.ok()) {
            return Status::runtime_error(
                "CUDA kernel " + std::string(op_type_name(op)) +
                " failed for node #" + std::to_string(nid.value) +
                " (" + std::string((*nd)->name()) + "): " + st.to_string());
        }
    }
    auto st = ctx.advance_kv_cache_step();
    if (!st.ok()) {
        return Status::runtime_error("KV cache advance failed: " + st.to_string());
    }
    return Status::make_ok();
}

#endif

} // namespace minillm
