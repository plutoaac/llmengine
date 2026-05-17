#pragma once

#include <expected>
#include <string>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/graph/node.h"
#include "minillm/graph/value.h"

namespace minillm {

class Graph {
public:
    std::expected<ValueId, Status> add_value(
        std::string name, Shape shape, DType dtype,
        Device device, ValueKind kind);

    std::expected<NodeId, Status> add_node(
        OpType op_type, std::string name,
        std::vector<ValueId> inputs, std::vector<ValueId> outputs);

    std::expected<Value*, Status> mutable_value(ValueId id);
    std::expected<const Value*, Status> value(ValueId id) const;

    std::expected<Node*, Status> mutable_node(NodeId id);
    std::expected<const Node*, Status> node(NodeId id) const;

    const std::vector<Value>& values() const;
    const std::vector<Node>& nodes() const;

    Status validate() const;
    std::expected<std::vector<NodeId>, Status> topological_sort() const;

    std::string dump() const;

private:
    std::vector<Value> values_;
    std::vector<Node> nodes_;
};

} // namespace minillm
