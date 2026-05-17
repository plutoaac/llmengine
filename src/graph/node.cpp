#include "minillm/graph/node.h"

namespace minillm {

Node::Node(NodeId id, OpType op_type, std::string name,
           std::vector<ValueId> inputs, std::vector<ValueId> outputs)
    : id_(id), op_type_(op_type), name_(std::move(name)),
      inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

NodeId Node::id() const { return id_; }
OpType Node::op_type() const { return op_type_; }
std::string_view Node::name() const { return name_; }

const std::vector<ValueId>& Node::inputs() const { return inputs_; }
const std::vector<ValueId>& Node::outputs() const { return outputs_; }

void Node::set_attr(std::string key, AttributeValue value) {
    attrs_[std::move(key)] = std::move(value);
}

std::optional<AttributeValue> Node::get_attr(std::string_view key) const {
    auto it = attrs_.find(std::string(key));
    if (it != attrs_.end()) return it->second;
    return std::nullopt;
}

const std::unordered_map<std::string, AttributeValue>& Node::attrs() const {
    return attrs_;
}

} // namespace minillm
