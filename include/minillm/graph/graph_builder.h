#pragma once

#include <expected>
#include <optional>
#include <string>

#include "minillm/core/device.h"
#include "minillm/core/dtype.h"
#include "minillm/core/status.h"
#include "minillm/graph/value.h"

namespace minillm {

class Graph;

class GraphBuilder {
public:
    explicit GraphBuilder(Graph& graph);

    std::expected<ValueId, Status> input(
        std::string name, Shape shape, DType dtype,
        Device device = Device::cpu());

    std::expected<ValueId, Status> constant(
        std::string name, Shape shape, DType dtype,
        Device device = Device::cpu());

    std::expected<ValueId, Status> embedding(
        ValueId input_ids, ValueId weight, std::string name);

    std::expected<ValueId, Status> matmul(
        ValueId a, ValueId b, std::string name);

    std::expected<ValueId, Status> linear(
        ValueId x, ValueId weight, std::optional<ValueId> bias,
        std::string name);

    std::expected<ValueId, Status> add(
        ValueId a, ValueId b, std::string name);

    std::expected<ValueId, Status> mul(
        ValueId a, ValueId b, std::string name);

    std::expected<ValueId, Status> rms_norm(
        ValueId x, ValueId weight, double eps, std::string name);

    std::expected<ValueId, Status> silu(
        ValueId x, std::string name);

    std::expected<ValueId, Status> swiglu(
        ValueId gate, ValueId up, std::string name);

    std::expected<ValueId, Status> rope(
        ValueId x, int64_t num_heads, int64_t head_dim,
        std::string name);

    std::expected<ValueId, Status> attention(
        ValueId q, ValueId k, ValueId v, bool causal,
        int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
        std::string name);

    std::expected<ValueId, Status> qk_norm(
        ValueId x, ValueId weight, double eps,
        int64_t num_heads, int64_t head_dim,
        std::string name);

    std::expected<ValueId, Status> output(
        ValueId x, std::string name);

private:
    Graph& graph_;

    std::expected<const Value*, Status> get_value(ValueId id) const;
    std::expected<ValueId, Status> make_intermediate(
        std::string name, Shape shape, DType dtype, Device device);
};

} // namespace minillm
