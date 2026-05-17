#include <cassert>
#include <iostream>

#include "minillm/graph/graph.h"
#include "minillm/graph/op_type.h"

using namespace minillm;

void test_add_input_value() {
    Graph g;
    auto vid = g.add_value("input", Shape({1, -1}), DType::Int32,
                           Device::cpu(), ValueKind::Input);
    assert(vid && vid->value == 0);
    auto v = g.value(*vid);
    assert(v && (*v)->name == "input");
    assert((*v)->kind == ValueKind::Input);
    std::cout << "  PASS test_add_input_value\n";
}

void test_add_constant_value() {
    Graph g;
    auto vid = g.add_value("weight", Shape({768, 768}), DType::Float32,
                           Device::cpu(), ValueKind::Constant);
    assert(vid && vid->value == 0);
    auto v = g.value(*vid);
    assert(v && (*v)->kind == ValueKind::Constant);
    std::cout << "  PASS test_add_constant_value\n";
}

void test_add_node() {
    Graph g;
    auto v0 = g.add_value("a", Shape({2, 3}), DType::Float32, Device::cpu(), ValueKind::Input);
    auto v1 = g.add_value("b", Shape({3, 4}), DType::Float32, Device::cpu(), ValueKind::Input);
    auto v2 = g.add_value("c", Shape({2, 4}), DType::Float32, Device::cpu(), ValueKind::Intermediate);
    assert(v0 && v1 && v2);
    auto nid = g.add_node(OpType::MatMul, "mm", {*v0, *v1}, {*v2});
    assert(nid && nid->value == 0);
    auto nd = g.node(*nid);
    assert(nd && (*nd)->op_type() == OpType::MatMul);
    assert((*nd)->inputs().size() == 2);
    std::cout << "  PASS test_add_node\n";
}

void test_invalid_valueid_validate_fails() {
    Graph g;
    auto v0 = g.add_value("a", Shape({2, 3}), DType::Float32, Device::cpu(), ValueKind::Input);
    assert(v0);
    // Node references a ValueId that doesn't exist
    ValueId bad{999};
    auto nid = g.add_node(OpType::MatMul, "mm", {*v0, bad}, {*v0});
    assert(nid);
    auto st = g.validate();
    assert(!st.ok());
    std::cout << "  PASS test_invalid_valueid_validate_fails\n";
}

void test_graph_dump_contains_op_name() {
    Graph g;
    auto v0 = g.add_value("a", Shape({2, 3}), DType::Float32, Device::cpu(), ValueKind::Input);
    auto v1 = g.add_value("b", Shape({2, 3}), DType::Float32, Device::cpu(), ValueKind::Input);
    auto v2 = g.add_value("c", Shape({2, 3}), DType::Float32, Device::cpu(), ValueKind::Intermediate);
    g.add_node(OpType::Add, "add1", {*v0, *v1}, {*v2});
    auto dump = g.dump();
    assert(dump.find("Add") != std::string::npos);
    std::cout << "  PASS test_graph_dump_contains_op_name\n";
}

int main() {
    std::cout << "test_graph:\n";
    test_add_input_value();
    test_add_constant_value();
    test_add_node();
    test_invalid_valueid_validate_fails();
    test_graph_dump_contains_op_name();
    std::cout << "All tests passed!\n";
    return 0;
}
