#include "minillm/graph/graph_builder.h"

#include "minillm/graph/graph.h"
#include "minillm/graph/shape_infer.h"

namespace minillm {

GraphBuilder::GraphBuilder(Graph& graph) : graph_(graph) {}

std::expected<const Value*, Status> GraphBuilder::get_value(ValueId id) const {
    return graph_.value(id);
}

std::expected<ValueId, Status> GraphBuilder::make_intermediate(
    std::string name, Shape shape, DType dtype, Device device) {
    return graph_.add_value(std::move(name), std::move(shape), dtype, device,
                            ValueKind::Intermediate);
}

std::expected<ValueId, Status> GraphBuilder::input(
    std::string name, Shape shape, DType dtype, Device device) {
    return graph_.add_value(std::move(name), std::move(shape), dtype, device,
                            ValueKind::Input);
}

std::expected<ValueId, Status> GraphBuilder::constant(
    std::string name, Shape shape, DType dtype, Device device) {
    return graph_.add_value(std::move(name), std::move(shape), dtype, device,
                            ValueKind::Constant);
}

std::expected<ValueId, Status> GraphBuilder::embedding(
    ValueId input_ids, ValueId weight, std::string name) {
    auto iv = get_value(input_ids);
    if (!iv) return std::unexpected(iv.error());
    auto wv = get_value(weight);
    if (!wv) return std::unexpected(wv.error());
    auto out_shape = infer_embedding_shape((*iv)->shape, (*wv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*wv)->dtype, (*wv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Embedding, std::move(name),
                               std::vector<ValueId>{input_ids, weight},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::matmul(
    ValueId a, ValueId b, std::string name) {
    auto av = get_value(a);
    if (!av) return std::unexpected(av.error());
    auto bv = get_value(b);
    if (!bv) return std::unexpected(bv.error());
    auto out_shape = infer_matmul_shape((*av)->shape, (*bv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*av)->dtype, (*av)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::MatMul, std::move(name),
                               std::vector<ValueId>{a, b},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::linear(
    ValueId x, ValueId weight, std::optional<ValueId> bias, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto wv = get_value(weight);
    if (!wv) return std::unexpected(wv.error());
    auto out_shape = infer_linear_shape((*xv)->shape, (*wv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    if (bias) {
        auto bv = get_value(*bias);
        if (!bv) return std::unexpected(bv.error());
        if ((*bv)->shape.rank() != 1 || (*bv)->shape.dim(0) != (*wv)->shape.dim(0)) {
            return std::unexpected(Status::shape_mismatch(
                "Linear bias must have shape [" + std::to_string((*wv)->shape.dim(0)) +
                "], got " + (*bv)->shape.to_string()));
        }
    }
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*wv)->dtype, (*wv)->device);
    if (!out) return std::unexpected(out.error());
    std::vector<ValueId> inputs{x, weight};
    if (bias) inputs.push_back(*bias);
    auto nid = graph_.add_node(OpType::Linear, std::move(name),
                               std::move(inputs),
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::add(
    ValueId a, ValueId b, std::string name) {
    auto av = get_value(a);
    if (!av) return std::unexpected(av.error());
    auto bv = get_value(b);
    if (!bv) return std::unexpected(bv.error());
    auto out_shape = infer_add_shape((*av)->shape, (*bv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*av)->dtype, (*av)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Add, std::move(name),
                               std::vector<ValueId>{a, b},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::mul(
    ValueId a, ValueId b, std::string name) {
    auto av = get_value(a);
    if (!av) return std::unexpected(av.error());
    auto bv = get_value(b);
    if (!bv) return std::unexpected(bv.error());
    auto out_shape = infer_elementwise_shape((*av)->shape, (*bv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*av)->dtype, (*av)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Mul, std::move(name),
                               std::vector<ValueId>{a, b},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::rms_norm(
    ValueId x, ValueId weight, double eps, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_same_shape_unary((*xv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::RMSNorm, std::move(name),
                               std::vector<ValueId>{x, weight},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("eps", eps);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::silu(
    ValueId x, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_same_shape_unary((*xv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::SiLU, std::move(name),
                               std::vector<ValueId>{x},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::swiglu(
    ValueId gate, ValueId up, std::string name) {
    auto gv = get_value(gate);
    if (!gv) return std::unexpected(gv.error());
    auto uv = get_value(up);
    if (!uv) return std::unexpected(uv.error());
    auto out_shape = infer_elementwise_shape((*gv)->shape, (*uv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*gv)->dtype, (*gv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::SwiGLU, std::move(name),
                               std::vector<ValueId>{gate, up},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    return out;
}

std::expected<ValueId, Status> GraphBuilder::rope(
    ValueId x, int64_t num_heads, int64_t head_dim, double rope_base, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_same_shape_unary((*xv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::RoPE, std::move(name),
                               std::vector<ValueId>{x},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("num_heads", num_heads);
        (*nd)->set_attr("head_dim", head_dim);
        (*nd)->set_attr("rope_base", rope_base);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::attention(
    ValueId q, ValueId k, ValueId v, bool causal,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t layer_idx,
    std::string name) {
    auto qv = get_value(q);
    if (!qv) return std::unexpected(qv.error());
    auto kv = get_value(k);
    if (!kv) return std::unexpected(kv.error());
    auto vv = get_value(v);
    if (!vv) return std::unexpected(vv.error());
    auto out = make_intermediate(name + ".out", (*qv)->shape,
                                 (*qv)->dtype, (*qv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Attention, std::move(name),
                               std::vector<ValueId>{q, k, v},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("causal", causal);
        (*nd)->set_attr("num_heads", num_heads);
        (*nd)->set_attr("num_kv_heads", num_kv_heads);
        (*nd)->set_attr("head_dim", head_dim);
        (*nd)->set_attr("layer_idx", layer_idx);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::qk_norm(
    ValueId x, ValueId weight, double eps,
    int64_t num_heads, int64_t head_dim, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto wv = get_value(weight);
    if (!wv) return std::unexpected(wv.error());
    auto out_shape = infer_same_shape_unary((*xv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::QKNorm, std::move(name),
                               std::vector<ValueId>{x, weight},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("eps", eps);
        (*nd)->set_attr("num_heads", num_heads);
        (*nd)->set_attr("head_dim", head_dim);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::softmax(
    ValueId x, int64_t axis, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_same_shape_unary((*xv)->shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Softmax, std::move(name),
                               std::vector<ValueId>{x},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("axis", axis);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::reshape(
    ValueId x, Shape target_shape, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_reshape_shape((*xv)->shape, target_shape);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", *out_shape,
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Reshape, std::move(name),
                               std::vector<ValueId>{x},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("shape", *out_shape);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::transpose(
    ValueId x, int64_t axis0, int64_t axis1, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto out_shape = infer_transpose_shape((*xv)->shape, axis0, axis1);
    if (!out_shape) return std::unexpected(out_shape.error());
    auto out = make_intermediate(name + ".out", std::move(*out_shape),
                                 (*xv)->dtype, (*xv)->device);
    if (!out) return std::unexpected(out.error());
    auto nid = graph_.add_node(OpType::Transpose, std::move(name),
                               std::vector<ValueId>{x},
                               std::vector<ValueId>{*out});
    if (!nid) return std::unexpected(nid.error());
    if (auto nd = graph_.mutable_node(*nid); nd) {
        (*nd)->set_attr("axis0", axis0);
        (*nd)->set_attr("axis1", axis1);
    }
    return out;
}

std::expected<ValueId, Status> GraphBuilder::output(
    ValueId x, std::string name) {
    auto xv = get_value(x);
    if (!xv) return std::unexpected(xv.error());
    auto val = graph_.mutable_value(x);
    if (!val) return std::unexpected(val.error());
    (*val)->kind = ValueKind::Output;
    return x;
}

} // namespace minillm
