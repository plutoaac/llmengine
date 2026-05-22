#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "minillm/minillm.h"

using namespace minillm;

void test_reuses_non_overlapping_intermediates() {
    Graph g;
    auto input = g.add_value("input", Shape({16}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    auto a = g.add_value("a", Shape({16}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    auto b = g.add_value("b", Shape({32}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    auto c = g.add_value("c", Shape({16}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    assert(input && a && b && c);

    assert(g.add_node(OpType::SiLU, "make_a", {*input}, {*a}));
    assert(g.add_node(OpType::Linear, "make_b", {*a, *input}, {*b}));
    assert(g.add_node(OpType::SiLU, "make_c", {*b}, {*c}));
    assert(g.add_node(OpType::Output, "out", {*c}, {}));

    auto plan = MemoryPlanner::plan(g);
    assert(plan);

    assert(plan->eligible_value_count == 3);
    assert(plan->naive_bytes == 256);
    assert(plan->planned_bytes == 192);
    assert(plan->buffers.size() == 2);

    auto* a_range = plan->range_for(*a);
    auto* b_range = plan->range_for(*b);
    auto* c_range = plan->range_for(*c);
    assert(a_range && b_range && c_range);
    assert(a_range->buffer_id == c_range->buffer_id);
    assert(a_range->buffer_id != b_range->buffer_id);
    assert(a_range->first_node == 0 && a_range->last_node == 1);
    assert(b_range->first_node == 1 && b_range->last_node == 2);
    assert(c_range->first_node == 2 && c_range->last_node == 3);

    auto report = plan->report();
    assert(report.find("Graph memory plan") != std::string::npos);
    assert(report.find("reuse saving") != std::string::npos);
    std::cout << "  PASS test_reuses_non_overlapping_intermediates\n";
}

void test_skips_non_plannable_values() {
    Graph g;
    auto input = g.add_value("input", Shape({8}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    auto weight = g.add_value("weight", Shape({8, 8}), DType::Float32,
                              Device::cpu(), ValueKind::Constant);
    auto dynamic = g.add_value("dynamic", Shape({-1, 8}), DType::Float32,
                               Device::cpu(), ValueKind::Intermediate);
    auto gpu = g.add_value("gpu", Shape({8}), DType::Float32,
                           Device::cuda(0), ValueKind::Intermediate);
    auto out = g.add_value("out", Shape({8}), DType::Float32,
                           Device::cpu(), ValueKind::Output);
    assert(input && weight && dynamic && gpu && out);

    assert(g.add_node(OpType::Custom, "n0", {*input, *weight}, {*dynamic}));
    assert(g.add_node(OpType::Custom, "n1", {*dynamic}, {*gpu}));
    assert(g.add_node(OpType::Custom, "n2", {*gpu}, {*out}));

    auto plan = MemoryPlanner::plan(g);
    assert(plan);
    assert(plan->eligible_value_count == 0);
    assert(plan->skipped_value_count == 5);
    assert(plan->naive_bytes == 0);
    assert(plan->planned_bytes == 0);

    auto* dynamic_range = plan->range_for(*dynamic);
    auto* gpu_range = plan->range_for(*gpu);
    auto* out_range = plan->range_for(*out);
    assert(dynamic_range && dynamic_range->skip_reason == "dynamic shape");
    assert(gpu_range && gpu_range->skip_reason == "non-CPU device");
    assert(out_range && out_range->skip_reason == "kind=Output");
    std::cout << "  PASS test_skips_non_plannable_values\n";
}

void test_runtime_context_uses_planned_arena() {
    Graph g;
    auto input = g.add_value("input", Shape({16}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    auto a = g.add_value("a", Shape({16}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    auto b = g.add_value("b", Shape({32}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    auto c = g.add_value("c", Shape({16}), DType::Float32,
                         Device::cpu(), ValueKind::Intermediate);
    assert(input && a && b && c);

    assert(g.add_node(OpType::SiLU, "make_a", {*input}, {*a}));
    assert(g.add_node(OpType::Linear, "make_b", {*a, *input}, {*b}));
    assert(g.add_node(OpType::SiLU, "make_c", {*b}, {*c}));
    assert(g.add_node(OpType::Output, "out", {*c}, {}));

    RuntimeContext ctx;
    auto plan = ctx.allocate_intermediates_planned(g);
    assert(plan);
    assert(plan->planned_bytes == 192);

    Tensor* a_tensor = ctx.get(*a);
    Tensor* b_tensor = ctx.get(*b);
    Tensor* c_tensor = ctx.get(*c);
    assert(a_tensor && b_tensor && c_tensor);
    assert(a_tensor->data() == c_tensor->data());
    assert(a_tensor->data() != b_tensor->data());
    assert(reinterpret_cast<std::uintptr_t>(a_tensor->data()) % plan->alignment == 0);

    std::cout << "  PASS test_runtime_context_uses_planned_arena\n";
}

void test_include_outputs_option() {
    Graph g;
    auto input = g.add_value("input", Shape({4}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    auto out = g.add_value("out", Shape({4}), DType::Float32,
                           Device::cpu(), ValueKind::Output);
    assert(input && out);
    assert(g.add_node(OpType::SiLU, "make_out", {*input}, {*out}));

    auto default_plan = MemoryPlanner::plan(g);
    assert(default_plan);
    assert(default_plan->eligible_value_count == 0);

    MemoryPlanOptions options;
    options.include_outputs = true;
    auto output_plan = MemoryPlanner::plan(g, options);
    assert(output_plan);
    assert(output_plan->eligible_value_count == 1);
    assert(output_plan->planned_bytes == 64);

    auto* out_range = output_plan->range_for(*out);
    assert(out_range && out_range->eligible);
    assert(out_range->first_node == 0);
    assert(out_range->last_node == 0);
    std::cout << "  PASS test_include_outputs_option\n";
}

void test_rejects_invalid_alignment() {
    Graph g;
    auto input = g.add_value("input", Shape({4}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    assert(input);

    MemoryPlanOptions options;
    options.alignment = 0;
    auto plan = MemoryPlanner::plan(g, options);
    assert(!plan);
    assert(plan.error().code() == ErrorCode::InvalidArgument);
    std::cout << "  PASS test_rejects_invalid_alignment\n";
}

int main() {
    std::cout << "test_memory_planner:\n";
    test_reuses_non_overlapping_intermediates();
    test_skips_non_plannable_values();
    test_runtime_context_uses_planned_arena();
    test_include_outputs_option();
    test_rejects_invalid_alignment();
    std::cout << "All tests passed!\n";
    return 0;
}
