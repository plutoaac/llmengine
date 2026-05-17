#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "minillm/graph/attribute.h"
#include "minillm/graph/op_type.h"
#include "minillm/graph/value.h"

namespace minillm {

class Node {
public:
    Node(NodeId id, OpType op_type, std::string name,
         std::vector<ValueId> inputs, std::vector<ValueId> outputs);

    NodeId id() const;
    OpType op_type() const;
    std::string_view name() const;

    const std::vector<ValueId>& inputs() const;
    const std::vector<ValueId>& outputs() const;

    void set_attr(std::string key, AttributeValue value);
    std::optional<AttributeValue> get_attr(std::string_view key) const;

    const std::unordered_map<std::string, AttributeValue>& attrs() const;

private:
    NodeId id_;
    OpType op_type_;
    std::string name_;
    std::vector<ValueId> inputs_;
    std::vector<ValueId> outputs_;
    std::unordered_map<std::string, AttributeValue> attrs_;
};

} // namespace minillm
