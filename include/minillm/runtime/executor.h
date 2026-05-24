#pragma once

#include <memory>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/graph/value.h"

namespace minillm {

class Backend;
class Graph;
class KernelRegistry;
class RuntimeContext;

class Executor {
public:
    virtual ~Executor() = default;
    virtual Status compile(const Graph& graph) = 0;
    virtual Status run(RuntimeContext& ctx) = 0;
};

class CpuExecutor final : public Executor {
public:
    CpuExecutor(std::shared_ptr<Backend> backend, KernelRegistry& registry);

    Status compile(const Graph& graph) override;
    Status run(RuntimeContext& ctx) override;

private:
    const Graph* graph_{nullptr};
    std::shared_ptr<Backend> backend_;
    KernelRegistry& registry_;
    std::vector<NodeId> execution_order_;
};

#if defined(MINILLM_ENABLE_CUDA)
class CudaExecutor final : public Executor {
public:
    CudaExecutor(std::shared_ptr<Backend> backend, KernelRegistry& registry);

    Status compile(const Graph& graph) override;
    Status run(RuntimeContext& ctx) override;

private:
    const Graph* graph_{nullptr};
    std::shared_ptr<Backend> backend_;
    KernelRegistry& registry_;
    std::vector<NodeId> execution_order_;
};
#endif

} // namespace minillm
