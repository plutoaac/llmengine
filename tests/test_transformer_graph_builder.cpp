#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "minillm/graph/graph.h"
#include "minillm/graph/graph_builder.h"
#include "minillm/model/transformer_graph_builder.h"

using namespace minillm;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) \
    do { \
        ++tests_run; \
        if (cond) { ++tests_passed; } \
        else { std::cerr << "FAIL: " << msg << " at line " << __LINE__ << "\n"; } \
    } while(0)

void test_build_tiny_decoder_block_weight_names() {
    Graph graph;
    GraphBuilder builder(graph);
    TransformerGraphBuilder tf_builder(builder);

    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 8;
    cfg.vocab_size = 100;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.num_layers = 1;

    auto result = tf_builder.build_tiny_decoder_block(cfg);
    CHECK(result.has_value(), "build_tiny_decoder_block should succeed");

    // Verify per-layer weight names have layer_0. prefix
    std::vector<std::string> expected_layer_weights = {
        "layer_0.attn_norm.weight",
        "layer_0.q_proj.weight",
        "layer_0.k_proj.weight",
        "layer_0.v_proj.weight",
        "layer_0.o_proj.weight",
        "layer_0.ffn_norm.weight",
        "layer_0.gate_proj.weight",
        "layer_0.up_proj.weight",
        "layer_0.down_proj.weight",
    };

    for (const auto& expected : expected_layer_weights) {
        bool found = false;
        for (const auto& v : graph.values()) {
            if (v.name == expected) { found = true; break; }
        }
        CHECK(found, "expected weight name: " + expected);
    }

    // Global weights should NOT have layer_0. prefix
    std::vector<std::string> expected_global_weights = {
        "tok_embeddings.weight",
        "lm_head.weight",
    };

    for (const auto& expected : expected_global_weights) {
        bool found = false;
        for (const auto& v : graph.values()) {
            if (v.name == expected) { found = true; break; }
        }
        CHECK(found, "expected global weight name: " + expected);
    }
}

void test_build_transformer_weight_names() {
    Graph graph;
    GraphBuilder builder(graph);
    TransformerGraphBuilder tf_builder(builder);

    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 8;
    cfg.vocab_size = 100;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.num_layers = 3;

    auto result = tf_builder.build_transformer(cfg);
    CHECK(result.has_value(), "build_transformer should succeed");

    // Verify multi-layer weight names
    for (int layer = 0; layer < 3; ++layer) {
        std::string prefix = "layer_" + std::to_string(layer) + ".";
        std::vector<std::string> expected = {
            prefix + "attn_norm.weight",
            prefix + "q_proj.weight",
            prefix + "k_proj.weight",
            prefix + "v_proj.weight",
            prefix + "o_proj.weight",
            prefix + "ffn_norm.weight",
            prefix + "gate_proj.weight",
            prefix + "up_proj.weight",
            prefix + "down_proj.weight",
        };

        for (const auto& name : expected) {
            bool found = false;
            for (const auto& v : graph.values()) {
                if (v.name == name) { found = true; break; }
            }
            CHECK(found, "expected layer weight: " + name);
        }
    }

    // Verify global weights
    for (const auto& name : {"tok_embeddings.weight", "lm_head.weight", "output_norm.weight"}) {
        bool found = false;
        for (const auto& v : graph.values()) {
            if (v.name == name) { found = true; break; }
        }
        CHECK(found, std::string("expected global weight: ") + name);
    }
}

void test_rope_base_attribute() {
    Graph graph;
    GraphBuilder builder(graph);
    TransformerGraphBuilder tf_builder(builder);

    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 4;
    cfg.vocab_size = 50;
    cfg.hidden_size = 32;
    cfg.num_heads = 2;
    cfg.head_dim = 16;
    cfg.num_layers = 1;
    cfg.rope_base = 500000.0;

    auto result = tf_builder.build_transformer(cfg);
    CHECK(result.has_value(), "build_transformer with custom rope_base should succeed");

    // Check that rope nodes have rope_base attribute
    bool found_rope_with_base = false;
    for (const auto& node : graph.nodes()) {
        if (node.op_type() == OpType::RoPE) {
            auto rb_attr = node.get_attr("rope_base");
            if (rb_attr) {
                auto* rb = std::get_if<double>(&*rb_attr);
                if (rb && *rb == 500000.0) {
                    found_rope_with_base = true;
                    break;
                }
            }
        }
    }
    CHECK(found_rope_with_base, "rope node should have rope_base attribute with custom value");
}

int main() {
    test_build_tiny_decoder_block_weight_names();
    test_build_transformer_weight_names();
    test_rope_base_attribute();

    std::cout << "TransformerGraphBuilder tests: " << tests_passed << "/" << tests_run << " passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
