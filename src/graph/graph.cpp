#include "minillm/graph/graph.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "minillm/core/dtype.h"

namespace minillm {

std::expected<ValueId, Status> Graph::add_value(
    std::string name, Shape shape, DType dtype, Device device, ValueKind kind) {
    size_t idx = values_.size();
    Value v{ValueId{idx}, std::move(name), std::move(shape), dtype, device, kind};
    values_.push_back(std::move(v));
    return ValueId{idx};
}

std::expected<NodeId, Status> Graph::add_node(
    OpType op_type, std::string name,
    std::vector<ValueId> inputs, std::vector<ValueId> outputs) {
    size_t idx = nodes_.size();
    nodes_.emplace_back(NodeId{idx}, op_type, std::move(name),
                        std::move(inputs), std::move(outputs));
    return NodeId{idx};
}

std::expected<Value*, Status> Graph::mutable_value(ValueId id) {
    if (id.value >= values_.size()) {
        return std::unexpected(Status::out_of_range(
            "ValueId %" + std::to_string(id.value) + " out of range"));
    }
    return &values_[id.value];
}

std::expected<const Value*, Status> Graph::value(ValueId id) const {
    if (id.value >= values_.size()) {
        return std::unexpected(Status::out_of_range(
            "ValueId %" + std::to_string(id.value) + " out of range"));
    }
    return &values_[id.value];
}

std::expected<Node*, Status> Graph::mutable_node(NodeId id) {
    if (id.value >= nodes_.size()) {
        return std::unexpected(Status::out_of_range(
            "NodeId #" + std::to_string(id.value) + " out of range"));
    }
    return &nodes_[id.value];
}

std::expected<const Node*, Status> Graph::node(NodeId id) const {
    if (id.value >= nodes_.size()) {
        return std::unexpected(Status::out_of_range(
            "NodeId #" + std::to_string(id.value) + " out of range"));
    }
    return &nodes_[id.value];
}

const std::vector<Value>& Graph::values() const { return values_; }
const std::vector<Node>& Graph::nodes() const { return nodes_; }

Status Graph::validate() const {
    // Check all Node inputs/outputs exist
    for (const auto& n : nodes_) {
        for (const auto& vid : n.inputs()) {
            if (vid.value >= values_.size()) {
                return Status::invalid_argument(
                    "Node #" + std::to_string(n.id().value) + " (" +
                    std::string(op_type_name(n.op_type())) +
                    ") references non-existent ValueId %" +
                    std::to_string(vid.value));
            }
        }
        for (const auto& vid : n.outputs()) {
            if (vid.value >= values_.size()) {
                return Status::invalid_argument(
                    "Node #" + std::to_string(n.id().value) + " (" +
                    std::string(op_type_name(n.op_type())) +
                    ") references non-existent ValueId %" +
                    std::to_string(vid.value));
            }
        }
        // Output nodes must have at least one input
        if (n.op_type() == OpType::Output && n.inputs().empty()) {
            return Status::invalid_argument(
                "Output node #" + std::to_string(n.id().value) +
                " must have at least one input");
        }
    }
    return Status::make_ok();
}

std::expected<std::vector<NodeId>, Status> Graph::topological_sort() const {
    auto st = validate();
    if (!st.ok()) return std::unexpected(st);

    // Build adjacency: node -> set of nodes it depends on (via inputs)
    size_t n = nodes_.size();
    std::vector<int> in_degree(n, 0);
    // Map value -> node that produces it
    std::unordered_map<size_t, size_t> value_producer;
    for (size_t i = 0; i < n; ++i) {
        for (const auto& vid : nodes_[i].outputs()) {
            value_producer[vid.value] = i;
        }
    }
    // Build edges: producer_node -> consumer_node
    std::vector<std::vector<size_t>> adj(n);
    for (size_t i = 0; i < n; ++i) {
        for (const auto& vid : nodes_[i].inputs()) {
            auto it = value_producer.find(vid.value);
            if (it != value_producer.end()) {
                adj[it->second].push_back(i);
                ++in_degree[i];
            }
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < n; ++i) {
        if (in_degree[i] == 0) q.push(i);
    }

    std::vector<NodeId> order;
    order.reserve(n);
    while (!q.empty()) {
        size_t cur = q.front();
        q.pop();
        order.push_back(NodeId{cur});
        for (size_t nxt : adj[cur]) {
            if (--in_degree[nxt] == 0) q.push(nxt);
        }
    }

    if (order.size() != n) {
        return std::unexpected(Status::runtime_error("graph has cycles"));
    }
    return order;
}

static std::string_view kind_name(ValueKind kind) {
    switch (kind) {
    case ValueKind::Input:       return "Input";
    case ValueKind::Constant:    return "Constant";
    case ValueKind::Intermediate:return "Intermediate";
    case ValueKind::Output:      return "Output";
    default:                     std::unreachable();
    }
}

std::string Graph::dump() const {
    std::string s = "Graph:\n  Values:\n";
    for (const auto& v : values_) {
        s += "    %" + std::to_string(v.id.value) + " " + v.name +
             " shape=" + v.shape.to_string() +
             " dtype=" + std::string(dtype_name(v.dtype)) +
             " device=" + v.device.to_string() +
             " kind=" + std::string(kind_name(v.kind)) + "\n";
    }
    s += "  Nodes:\n";
    for (const auto& nd : nodes_) {
        s += "    #" + std::to_string(nd.id().value) + " " +
             std::string(op_type_name(nd.op_type())) +
             "(name=" + std::string(nd.name()) + ") inputs=[";
        for (size_t i = 0; i < nd.inputs().size(); ++i) {
            if (i > 0) s += ", ";
            s += "%" + std::to_string(nd.inputs()[i].value);
        }
        s += "] outputs=[";
        for (size_t i = 0; i < nd.outputs().size(); ++i) {
            if (i > 0) s += ", ";
            s += "%" + std::to_string(nd.outputs()[i].value);
        }
        s += "]";
        if (!nd.attrs().empty()) {
            s += " attrs={";
            bool first = true;
            for (const auto& [k, v] : nd.attrs()) {
                if (!first) s += ", ";
                first = false;
                s += k + "=" + attribute_to_string(v);
            }
            s += "}";
        }
        s += "\n";
    }
    return s;
}

} // namespace minillm
