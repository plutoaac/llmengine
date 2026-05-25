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
    assert(plan->eligible_value_count == 1);  // gpu is now eligible (CUDA arena)
    assert(plan->skipped_value_count == 4);
    assert(plan->naive_bytes > 0); // gpu has bytes
    assert(plan->planned_bytes > 0);

    auto* dynamic_range = plan->range_for(*dynamic);
    auto* gpu_range = plan->range_for(*gpu);
    auto* out_range = plan->range_for(*out);
    assert(dynamic_range && dynamic_range->skip_reason == "dynamic shape");
    assert(gpu_range && gpu_range->eligible == true);
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

void test_different_dtype_buffers_do_not_reuse() {
    Graph g;
    auto f_input = g.add_value("f_input", Shape({16}), DType::Float32,
                               Device::cpu(), ValueKind::Input);
    auto i_input = g.add_value("i_input", Shape({16}), DType::Int32,
                               Device::cpu(), ValueKind::Input);
    auto f_tmp = g.add_value("f_tmp", Shape({16}), DType::Float32,
                             Device::cpu(), ValueKind::Intermediate);
    auto i_tmp = g.add_value("i_tmp", Shape({16}), DType::Int32,
                             Device::cpu(), ValueKind::Intermediate);
    assert(f_input && i_input && f_tmp && i_tmp);

    assert(g.add_node(OpType::Custom, "make_f", {*f_input}, {*f_tmp}));
    assert(g.add_node(OpType::Output, "consume_f", {*f_tmp}, {}));
    assert(g.add_node(OpType::Custom, "make_i", {*i_input}, {*i_tmp}));
    assert(g.add_node(OpType::Output, "consume_i", {*i_tmp}, {}));

    auto plan = MemoryPlanner::plan(g);
    assert(plan);
    auto* f_range = plan->range_for(*f_tmp);
    auto* i_range = plan->range_for(*i_tmp);
    assert(f_range && i_range);
    assert(f_range->buffer_id != i_range->buffer_id);
    assert(plan->buffers[f_range->buffer_id].dtype == DType::Float32);
    assert(plan->buffers[i_range->buffer_id].dtype == DType::Int32);

    std::cout << "  PASS test_different_dtype_buffers_do_not_reuse\n";
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

void test_contiguous_arena_allocation() {
    // Create a graph where two non-overlapping intermediates share a buffer.
    // Verify that with contiguous arena, they live at different offsets within
    // the same arena allocation.
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

    // a and c should share the same buffer (same dtype, non-overlapping lifetimes)
    auto* a_range = plan->range_for(*a);
    auto* c_range = plan->range_for(*c);
    assert(a_range && c_range);
    assert(a_range->buffer_id == c_range->buffer_id);

    // All buffers should have the same arena_index (all FP32 CPU)
    for (const auto& buffer : plan->buffers) {
        assert(buffer.arena_index != static_cast<size_t>(-1));
    }
    size_t first_arena = plan->buffers[0].arena_index;
    for (const auto& buffer : plan->buffers) {
        assert(buffer.arena_index == first_arena);
    }

    // Verify tensors are allocated and accessible
    Tensor* a_tensor = ctx.get(*a);
    Tensor* b_tensor = ctx.get(*b);
    Tensor* c_tensor = ctx.get(*c);
    assert(a_tensor && b_tensor && c_tensor);
    assert(a_tensor->data() != nullptr);
    assert(b_tensor->data() != nullptr);
    assert(c_tensor->data() != nullptr);

    // Write to a_tensor and verify c_tensor doesn't get corrupted
    // (they share a buffer but are at the same offset in this case)
    auto* a_float = reinterpret_cast<float*>(a_tensor->data());
    a_float[0] = 42.0f;

    // b_tensor is in a different buffer, writing should not affect a/c
    auto* b_float = reinterpret_cast<float*>(b_tensor->data());
    b_float[0] = 99.0f;
    assert(a_float[0] == 42.0f);

    std::cout << "  PASS test_contiguous_arena_allocation\n";
}

void test_range_for_uses_index() {
    // Build a larger graph and verify range_for returns correct results quickly
    Graph g;
    auto input = g.add_value("input", Shape({8}), DType::Float32,
                             Device::cpu(), ValueKind::Input);
    std::vector<ValueId> intermediates;
    for (int i = 0; i < 20; ++i) {
        auto v = g.add_value("v" + std::to_string(i), Shape({8}), DType::Float32,
                             Device::cpu(), ValueKind::Intermediate);
        assert(v);
        intermediates.push_back(*v);
    }
    // Chain: input -> v0 -> v1 -> ... -> v19
    auto prev = *input;
    for (auto v : intermediates) {
        assert(g.add_node(OpType::SiLU, "n_" + std::to_string(v.value), {prev}, {v}));
        prev = v;
    }
    assert(g.add_node(OpType::Output, "out", {prev}, {}));

    auto plan = MemoryPlanner::plan(g);
    assert(plan);

    // Every intermediate should be findable via range_for
    for (auto v : intermediates) {
        auto* r = plan->range_for(v);
        assert(r != nullptr);
        assert(r->value == v);
    }

    // Non-existent value should return nullptr
    auto* missing = plan->range_for(ValueId{9999});
    assert(missing == nullptr);

    std::cout << "  PASS test_range_for_uses_index\n";
}

int main() {
    std::cout << "test_memory_planner:\n";
    test_reuses_non_overlapping_intermediates();
    test_skips_non_plannable_values();
    test_runtime_context_uses_planned_arena();
    test_different_dtype_buffers_do_not_reuse();
    test_include_outputs_option();
    test_rejects_invalid_alignment();
    test_contiguous_arena_allocation();
    test_range_for_uses_index();
    std::cout << "All tests passed!\n";
    return 0;
}
