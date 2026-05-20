#include <cassert>
#include <iostream>

#include "minillm/graph/graph.h"
#include "minillm/graph/graph_builder.h"
#include "minillm/core/shape.h"
#include "minillm/core/dtype.h"
#include "minillm/core/device.h"
#include "minillm/model/transformer_graph_builder.h"

using namespace minillm;

void test_embedding_shape_inference() {
    Graph g;
    GraphBuilder gb(g);
    auto ids = gb.input("ids", Shape({1, -1}), DType::Int32);
    assert(ids);
    auto w = gb.constant("w", Shape({32000, 768}), DType::Float32);
    assert(w);
    auto out = gb.embedding(*ids, *w, "emb");
    assert(out);
    auto v = g.value(*out);
    assert(v);
    assert((*v)->shape == Shape({1, -1, 768}));
    std::cout << "  PASS test_embedding_shape_inference\n";
}

void test_linear_shape_inference() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({1, -1, 768}), DType::Float32);
    assert(x);
    auto w = gb.constant("w", Shape({768, 768}), DType::Float32);
    assert(w);
    auto out = gb.linear(*x, *w, std::nullopt, "linear");
    assert(out);
    auto v = g.value(*out);
    assert(v);
    assert((*v)->shape == Shape({1, -1, 768}));
    std::cout << "  PASS test_linear_shape_inference\n";
}

void test_linear_shape_mismatch() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({1, -1, 512}), DType::Float32);
    assert(x);
    auto w = gb.constant("w", Shape({768, 256}), DType::Float32);
    assert(w);
    auto out = gb.linear(*x, *w, std::nullopt, "linear");
    assert(!out);
    assert(out.error().code() == ErrorCode::ShapeMismatch);
    std::cout << "  PASS test_linear_shape_mismatch\n";
}

void test_linear_bias_shape_mismatch() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({1, 2, 4}), DType::Float32);
    assert(x);
    auto w = gb.constant("w", Shape({3, 4}), DType::Float32);
    assert(w);
    auto b = gb.constant("b", Shape({2}), DType::Float32);
    assert(b);

    auto out = gb.linear(*x, *w, *b, "linear");
    assert(!out);
    assert(out.error().code() == ErrorCode::ShapeMismatch);
    std::cout << "  PASS test_linear_bias_shape_mismatch\n";
}

void test_add_shape_match() {
    Graph g;
    GraphBuilder gb(g);
    auto a = gb.input("a", Shape({1, -1, 768}), DType::Float32);
    assert(a);
    auto b = gb.input("b", Shape({1, -1, 768}), DType::Float32);
    assert(b);
    auto out = gb.add(*a, *b, "add");
    assert(out);
    auto v = g.value(*out);
    assert(v);
    assert((*v)->shape == Shape({1, -1, 768}));
    std::cout << "  PASS test_add_shape_match\n";
}

void test_add_shape_mismatch() {
    Graph g;
    GraphBuilder gb(g);
    auto a = gb.input("a", Shape({1, -1, 768}), DType::Float32);
    assert(a);
    auto b = gb.input("b", Shape({1, -1, 512}), DType::Float32);
    assert(b);
    auto out = gb.add(*a, *b, "add");
    assert(!out);
    assert(out.error().code() == ErrorCode::ShapeMismatch);
    std::cout << "  PASS test_add_shape_mismatch\n";
}

void test_tiny_decoder_graph_validate() {
    Graph g;
    GraphBuilder gb(g);
    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = -1;
    cfg.vocab_size = 32000;
    cfg.hidden_size = 768;
    cfg.intermediate_size = 2048;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 12;
    cfg.head_dim = 64;
    cfg.rms_norm_eps = 1e-6;

    TransformerGraphBuilder tb(gb);
    auto result = tb.build_tiny_decoder_block(cfg);
    assert(result);
    auto st = g.validate();
    assert(st.ok());
    std::cout << "  PASS test_tiny_decoder_graph_validate\n";
}

void test_final_logits_shape() {
    Graph g;
    GraphBuilder gb(g);
    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = -1;
    cfg.vocab_size = 32000;
    cfg.hidden_size = 768;
    cfg.intermediate_size = 2048;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 12;
    cfg.head_dim = 64;
    cfg.rms_norm_eps = 1e-6;

    TransformerGraphBuilder tb(gb);
    auto result = tb.build_tiny_decoder_block(cfg);
    assert(result);

    // The output value should have shape [1, -1, 32000]
    auto out_val = g.value(*result);
    assert(out_val);
    assert((*out_val)->shape == Shape({1, -1, 32000}));
    std::cout << "  PASS test_final_logits_shape\n";
}

void test_softmax_reshape_transpose_shapes() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({2, 3, 4}), DType::Float32);
    assert(x);

    auto probs = gb.softmax(*x, -1, "softmax");
    assert(probs);
    auto probs_v = g.value(*probs);
    assert(probs_v);
    assert((*probs_v)->shape == Shape({2, 3, 4}));

    auto flat = gb.reshape(*probs, Shape({6, -1}), "reshape");
    assert(flat);
    auto flat_v = g.value(*flat);
    assert(flat_v);
    assert((*flat_v)->shape == Shape({6, 4}));

    auto transposed = gb.transpose(*flat, 0, 1, "transpose");
    assert(transposed);
    auto transposed_v = g.value(*transposed);
    assert(transposed_v);
    assert((*transposed_v)->shape == Shape({4, 6}));

    std::cout << "  PASS test_softmax_reshape_transpose_shapes\n";
}

void test_reshape_element_mismatch() {
    Graph g;
    GraphBuilder gb(g);
    auto x = gb.input("x", Shape({2, 3, 4}), DType::Float32);
    assert(x);

    auto out = gb.reshape(*x, Shape({5, 5}), "bad_reshape");
    assert(!out);
    assert(out.error().code() == ErrorCode::ShapeMismatch);
    std::cout << "  PASS test_reshape_element_mismatch\n";
}

int main() {
    std::cout << "test_graph_builder:\n";
    test_embedding_shape_inference();
    test_linear_shape_inference();
    test_linear_shape_mismatch();
    test_linear_bias_shape_mismatch();
    test_add_shape_match();
    test_add_shape_mismatch();
    test_tiny_decoder_graph_validate();
    test_final_logits_shape();
    test_softmax_reshape_transpose_shapes();
    test_reshape_element_mismatch();
    std::cout << "All tests passed!\n";
    return 0;
}
