#include "minillm/runtime/executor.h"

#include <iostream>

#include "minillm/graph/graph.h"
#include "minillm/graph/op_type.h"
#include "minillm/runtime/backend.h"
#include "minillm/runtime/kernel_registry.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

// --- MockExecutor ---

MockExecutor::MockExecutor(std::shared_ptr<Backend> backend)
    : backend_(std::move(backend)) {}

Status MockExecutor::compile(const Graph& graph) {
    auto st = graph.validate();
    if (!st.ok()) return st;

    auto order = graph.topological_sort();
    if (!order) return order.error();

    for (const auto& nid : *order) {
        auto nd = graph.node(nid);
        if (!nd) return nd.error();
        if (!backend_->supports((*nd)->op_type())) {
            return Status::unsupported(
                "backend " + std::string(backend_->name()) +
                " does not support op " + std::string(op_type_name((*nd)->op_type())));
        }
    }

    graph_ = &graph;
    execution_order_ = std::move(*order);
    return Status::make_ok();
}

Status MockExecutor::run(RuntimeContext& /*ctx*/) {
    if (!graph_) {
        return Status::runtime_error("executor not compiled");
    }
    std::cout << "MockExecutor: running " << execution_order_.size() << " nodes\n";
    for (const auto& nid : execution_order_) {
        auto nd = graph_->node(nid);
        if (!nd) return nd.error();

        std::cout << "  Node #" << nid.value
                  << " " << op_type_name((*nd)->op_type())
                  << "(name=" << (*nd)->name() << ") inputs=[";
        for (size_t i = 0; i < (*nd)->inputs().size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "%" << (*nd)->inputs()[i].value;
        }
        std::cout << "] outputs=[";
        for (size_t i = 0; i < (*nd)->outputs().size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "%" << (*nd)->outputs()[i].value;
        }
        std::cout << "]\n";
    }
    return Status::make_ok();
}

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
    return Status::make_ok();
}

} // namespace minillm
