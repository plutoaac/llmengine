#include "minillm/model/transformer_graph_builder.h"

#include "minillm/core/device.h"
#include "minillm/graph/graph_builder.h"

namespace minillm {

TransformerGraphBuilder::TransformerGraphBuilder(GraphBuilder& builder)
    : builder_(builder) {}

std::expected<ValueId, Status> TransformerGraphBuilder::build_tiny_decoder_block(
    const TransformerConfig& cfg) {

    auto dev = cfg.device;

    // --- Inputs & weights (constants) ---
    auto input_ids = builder_.input("input_ids",
        Shape({cfg.batch_size, cfg.seq_len}), DType::Int32, dev);
    if (!input_ids) return std::unexpected(input_ids.error());

    auto tok_emb_w = builder_.constant("tok_embeddings.weight",
        Shape({cfg.vocab_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!tok_emb_w) return std::unexpected(tok_emb_w.error());

    auto attn_norm_w = builder_.constant("layer_0.attn_norm.weight",
        Shape({cfg.hidden_size}), cfg.dtype, dev);
    if (!attn_norm_w) return std::unexpected(attn_norm_w.error());

    auto q_proj_w = builder_.constant("layer_0.q_proj.weight",
        Shape({cfg.num_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
    if (!q_proj_w) return std::unexpected(q_proj_w.error());

    auto k_proj_w = builder_.constant("layer_0.k_proj.weight",
        Shape({cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
    if (!k_proj_w) return std::unexpected(k_proj_w.error());

    auto v_proj_w = builder_.constant("layer_0.v_proj.weight",
        Shape({cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
    if (!v_proj_w) return std::unexpected(v_proj_w.error());

    auto o_proj_w = builder_.constant("layer_0.o_proj.weight",
        Shape({cfg.hidden_size, cfg.num_heads * cfg.head_dim}), cfg.dtype, dev);
    if (!o_proj_w) return std::unexpected(o_proj_w.error());

    auto ffn_norm_w = builder_.constant("layer_0.ffn_norm.weight",
        Shape({cfg.hidden_size}), cfg.dtype, dev);
    if (!ffn_norm_w) return std::unexpected(ffn_norm_w.error());

    auto gate_proj_w = builder_.constant("layer_0.gate_proj.weight",
        Shape({cfg.intermediate_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!gate_proj_w) return std::unexpected(gate_proj_w.error());

    auto up_proj_w = builder_.constant("layer_0.up_proj.weight",
        Shape({cfg.intermediate_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!up_proj_w) return std::unexpected(up_proj_w.error());

    auto down_proj_w = builder_.constant("layer_0.down_proj.weight",
        Shape({cfg.hidden_size, cfg.intermediate_size}), cfg.dtype, dev);
    if (!down_proj_w) return std::unexpected(down_proj_w.error());

    auto lm_head_w = builder_.constant("lm_head.weight",
        Shape({cfg.vocab_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!lm_head_w) return std::unexpected(lm_head_w.error());

    // --- Forward pass ---
    auto hidden = builder_.embedding(*input_ids, *tok_emb_w, "tok_embedding");
    if (!hidden) return std::unexpected(hidden.error());

    auto residual1 = *hidden;

    auto normed = builder_.rms_norm(*hidden, *attn_norm_w, cfg.rms_norm_eps, "attn_norm");
    if (!normed) return std::unexpected(normed.error());

    auto q = builder_.linear(*normed, *q_proj_w, std::nullopt, "q_proj");
    if (!q) return std::unexpected(q.error());

    auto k = builder_.linear(*normed, *k_proj_w, std::nullopt, "k_proj");
    if (!k) return std::unexpected(k.error());

    auto v = builder_.linear(*normed, *v_proj_w, std::nullopt, "v_proj");
    if (!v) return std::unexpected(v.error());

    auto q_rope = builder_.rope(*q, cfg.num_heads, cfg.head_dim, cfg.rope_base, "rope_q");
    if (!q_rope) return std::unexpected(q_rope.error());

    auto k_rope = builder_.rope(*k, cfg.num_kv_heads, cfg.head_dim, cfg.rope_base, "rope_k");
    if (!k_rope) return std::unexpected(k_rope.error());

    auto attn_out = builder_.attention(*q_rope, *k_rope, *v, true,
                                       cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                                       int64_t(0), "attention");
    if (!attn_out) return std::unexpected(attn_out.error());

    auto o_proj = builder_.linear(*attn_out, *o_proj_w, std::nullopt, "o_proj");
    if (!o_proj) return std::unexpected(o_proj.error());

    auto add1 = builder_.add(*o_proj, residual1, "residual_add_1");
    if (!add1) return std::unexpected(add1.error());

    auto residual2 = *add1;

    auto ffn_normed = builder_.rms_norm(*add1, *ffn_norm_w, cfg.rms_norm_eps, "ffn_norm");
    if (!ffn_normed) return std::unexpected(ffn_normed.error());

    auto gate = builder_.linear(*ffn_normed, *gate_proj_w, std::nullopt, "gate_proj");
    if (!gate) return std::unexpected(gate.error());

    auto up = builder_.linear(*ffn_normed, *up_proj_w, std::nullopt, "up_proj");
    if (!up) return std::unexpected(up.error());

    auto swiglu_out = builder_.swiglu(*gate, *up, "swiglu");
    if (!swiglu_out) return std::unexpected(swiglu_out.error());

    auto down = builder_.linear(*swiglu_out, *down_proj_w, std::nullopt, "down_proj");
    if (!down) return std::unexpected(down.error());

    auto add2 = builder_.add(*down, residual2, "residual_add_2");
    if (!add2) return std::unexpected(add2.error());

    auto logits = builder_.linear(*add2, *lm_head_w, std::nullopt, "lm_head");
    if (!logits) return std::unexpected(logits.error());

    auto out = builder_.output(*logits, "output");
    if (!out) return std::unexpected(out.error());

    return out;
}

std::expected<ValueId, Status> TransformerGraphBuilder::build_transformer(
    const TransformerConfig& cfg) {

    auto dev = cfg.device;

    // --- Inputs ---
    auto input_ids = builder_.input("input_ids",
        Shape({cfg.batch_size, cfg.seq_len}), DType::Int32, dev);
    if (!input_ids) return std::unexpected(input_ids.error());

    // --- Shared weights ---
    auto tok_emb_w = builder_.constant("tok_embeddings.weight",
        Shape({cfg.vocab_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!tok_emb_w) return std::unexpected(tok_emb_w.error());

    auto lm_head_w = builder_.constant("lm_head.weight",
        Shape({cfg.vocab_size, cfg.hidden_size}), cfg.dtype, dev);
    if (!lm_head_w) return std::unexpected(lm_head_w.error());

    // --- Embedding ---
    auto hidden = builder_.embedding(*input_ids, *tok_emb_w, "tok_embedding");
    if (!hidden) return std::unexpected(hidden.error());

    // --- Transformer layers ---
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        std::string prefix = "layer_" + std::to_string(layer) + ".";

        // Per-layer weights
        auto attn_norm_w = builder_.constant(prefix + "attn_norm.weight",
            Shape({cfg.hidden_size}), cfg.dtype, dev);
        if (!attn_norm_w) return std::unexpected(attn_norm_w.error());

        auto q_proj_w = builder_.constant(prefix + "q_proj.weight",
            Shape({cfg.num_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
        if (!q_proj_w) return std::unexpected(q_proj_w.error());

        auto k_proj_w = builder_.constant(prefix + "k_proj.weight",
            Shape({cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
        if (!k_proj_w) return std::unexpected(k_proj_w.error());

        auto v_proj_w = builder_.constant(prefix + "v_proj.weight",
            Shape({cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size}), cfg.dtype, dev);
        if (!v_proj_w) return std::unexpected(v_proj_w.error());

        auto o_proj_w = builder_.constant(prefix + "o_proj.weight",
            Shape({cfg.hidden_size, cfg.num_heads * cfg.head_dim}), cfg.dtype, dev);
        if (!o_proj_w) return std::unexpected(o_proj_w.error());

        // Optional QK-norm weights (Qwen3-style)
        std::optional<ValueId> q_norm_w, k_norm_w;
        if (cfg.use_qk_norm) {
            auto qn = builder_.constant(prefix + "q_norm.weight",
                Shape({cfg.head_dim}), cfg.dtype, dev);
            if (!qn) return std::unexpected(qn.error());
            q_norm_w = *qn;

            auto kn = builder_.constant(prefix + "k_norm.weight",
                Shape({cfg.head_dim}), cfg.dtype, dev);
            if (!kn) return std::unexpected(kn.error());
            k_norm_w = *kn;
        }

        auto ffn_norm_w = builder_.constant(prefix + "ffn_norm.weight",
            Shape({cfg.hidden_size}), cfg.dtype, dev);
        if (!ffn_norm_w) return std::unexpected(ffn_norm_w.error());

        auto gate_proj_w = builder_.constant(prefix + "gate_proj.weight",
            Shape({cfg.intermediate_size, cfg.hidden_size}), cfg.dtype, dev);
        if (!gate_proj_w) return std::unexpected(gate_proj_w.error());

        auto up_proj_w = builder_.constant(prefix + "up_proj.weight",
            Shape({cfg.intermediate_size, cfg.hidden_size}), cfg.dtype, dev);
        if (!up_proj_w) return std::unexpected(up_proj_w.error());

        auto down_proj_w = builder_.constant(prefix + "down_proj.weight",
            Shape({cfg.hidden_size, cfg.intermediate_size}), cfg.dtype, dev);
        if (!down_proj_w) return std::unexpected(down_proj_w.error());

        // --- Attention block ---
        auto residual1 = *hidden;

        auto normed = builder_.rms_norm(*hidden, *attn_norm_w, cfg.rms_norm_eps,
                                        prefix + "attn_norm");
        if (!normed) return std::unexpected(normed.error());

        auto q = builder_.linear(*normed, *q_proj_w, std::nullopt, prefix + "q_proj");
        if (!q) return std::unexpected(q.error());

        auto k = builder_.linear(*normed, *k_proj_w, std::nullopt, prefix + "k_proj");
        if (!k) return std::unexpected(k.error());

        auto v = builder_.linear(*normed, *v_proj_w, std::nullopt, prefix + "v_proj");
        if (!v) return std::unexpected(v.error());

        // Optional QK-norm (RMSNorm per-head)
        if (cfg.use_qk_norm && q_norm_w && k_norm_w) {
            auto q_normed = builder_.qk_norm(*q, *q_norm_w, cfg.rms_norm_eps,
                                             cfg.num_heads, cfg.head_dim, prefix + "q_norm");
            if (!q_normed) return std::unexpected(q_normed.error());
            q = *q_normed;

            auto k_normed = builder_.qk_norm(*k, *k_norm_w, cfg.rms_norm_eps,
                                             cfg.num_kv_heads, cfg.head_dim, prefix + "k_norm");
            if (!k_normed) return std::unexpected(k_normed.error());
            k = *k_normed;
        }

        auto q_rope = builder_.rope(*q, cfg.num_heads, cfg.head_dim, cfg.rope_base, prefix + "rope_q");
        if (!q_rope) return std::unexpected(q_rope.error());

        auto k_rope = builder_.rope(*k, cfg.num_kv_heads, cfg.head_dim, cfg.rope_base, prefix + "rope_k");
        if (!k_rope) return std::unexpected(k_rope.error());

        auto attn_out = builder_.attention(*q_rope, *k_rope, *v, true,
                                           cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                                           layer,
                                           prefix + "attention");
        if (!attn_out) return std::unexpected(attn_out.error());

        auto o_proj = builder_.linear(*attn_out, *o_proj_w, std::nullopt, prefix + "o_proj");
        if (!o_proj) return std::unexpected(o_proj.error());

        auto add1 = builder_.add(*o_proj, residual1, prefix + "residual_add_1");
        if (!add1) return std::unexpected(add1.error());

        // --- FFN block ---
        auto residual2 = *add1;

        auto ffn_normed = builder_.rms_norm(*add1, *ffn_norm_w, cfg.rms_norm_eps,
                                            prefix + "ffn_norm");
        if (!ffn_normed) return std::unexpected(ffn_normed.error());

        auto gate = builder_.linear(*ffn_normed, *gate_proj_w, std::nullopt, prefix + "gate_proj");
        if (!gate) return std::unexpected(gate.error());

        auto up = builder_.linear(*ffn_normed, *up_proj_w, std::nullopt, prefix + "up_proj");
        if (!up) return std::unexpected(up.error());

        auto swiglu_out = builder_.swiglu(*gate, *up, prefix + "swiglu");
        if (!swiglu_out) return std::unexpected(swiglu_out.error());

        auto down = builder_.linear(*swiglu_out, *down_proj_w, std::nullopt, prefix + "down_proj");
        if (!down) return std::unexpected(down.error());

        auto add2 = builder_.add(*down, residual2, prefix + "residual_add_2");
        if (!add2) return std::unexpected(add2.error());

        hidden = *add2;
    }

    // --- Final norm + lm_head ---
    auto final_norm_w = builder_.constant("output_norm.weight",
        Shape({cfg.hidden_size}), cfg.dtype, dev);
    if (!final_norm_w) return std::unexpected(final_norm_w.error());

    auto final_normed = builder_.rms_norm(*hidden, *final_norm_w, cfg.rms_norm_eps, "output_norm");
    if (!final_normed) return std::unexpected(final_normed.error());

    auto logits = builder_.linear(*final_normed, *lm_head_w, std::nullopt, "lm_head");
    if (!logits) return std::unexpected(logits.error());

    auto out = builder_.output(*logits, "output");
    if (!out) return std::unexpected(out.error());

    return out;
}

} // namespace minillm
