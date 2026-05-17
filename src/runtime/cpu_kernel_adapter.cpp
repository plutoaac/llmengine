#include "minillm/runtime/cpu_kernel_adapter.h"

#include "minillm/core/dtype.h"
#include "minillm/core/shape.h"
#include "minillm/core/tensor.h"
#include "minillm/graph/node.h"
#include "minillm/graph/op_type.h"
#include "minillm/runtime/cpu_kernels.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

static Status check_dtype_float(const Tensor* t, std::string_view name) {
    if (t->dtype() != DType::Float32) {
        return Status::unsupported(
            std::string(name) + " only supports Float32, got " +
            std::string(dtype_name(t->dtype())));
    }
    return Status::make_ok();
}

static Status check_allocated(const Tensor* t, std::string_view name) {
    if (!t->is_allocated()) {
        return Status::runtime_error(
            std::string(name) + " tensor not allocated");
    }
    return Status::make_ok();
}

static const float* float_data(const Tensor* t) {
    return reinterpret_cast<const float*>(t->data());
}

static float* float_data_mut(Tensor* t) {
    return reinterpret_cast<float*>(t->data());
}

static const int* int_data(const Tensor* t) {
    return reinterpret_cast<const int*>(t->data());
}

// Helper to resolve ValueId to Tensor*, with error checks
static std::expected<Tensor*, Status> get_tensor(ValueId id, RuntimeContext& ctx,
                                                  std::string_view role) {
    auto* t = ctx.get(id);
    if (!t) {
        return std::unexpected(Status::runtime_error(
            std::string(role) + " tensor not found for ValueId %" +
            std::to_string(id.value)));
    }
    return t;
}

static Status kernel_embedding(const Node& node, RuntimeContext& ctx) {
    auto ids_t = get_tensor(node.inputs()[0], ctx, "input_ids");
    if (!ids_t) return ids_t.error();
    auto wt = get_tensor(node.inputs()[1], ctx, "weight");
    if (!wt) return wt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*ids_t, "input_ids"); if (!st.ok()) return st;
    st = check_allocated(*wt, "weight"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;
    st = check_dtype_float(*wt, "weight"); if (!st.ok()) return st;
    st = check_dtype_float(*ot, "output"); if (!st.ok()) return st;

    int seq_len = static_cast<int>((*ids_t)->shape().dim(1));
    int hidden = static_cast<int>((*wt)->shape().dim(1));
    cpu::embedding(float_data(*wt), int_data(*ids_t), float_data_mut(*ot),
                   seq_len, hidden);
    return Status::make_ok();
}

static Status kernel_linear(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto wt = get_tensor(node.inputs()[1], ctx, "weight");
    if (!wt) return wt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x"); if (!st.ok()) return st;
    st = check_allocated(*wt, "weight"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;
    st = check_dtype_float(*xt, "x"); if (!st.ok()) return st;
    st = check_dtype_float(*wt, "weight"); if (!st.ok()) return st;
    st = check_dtype_float(*ot, "output"); if (!st.ok()) return st;

    // x: [batch, seq, in] flattened to [M, K]
    // weight: [out, in] stored as [N, K]
    // y = x @ W^T => C = A @ B^T where B = W
    int M = 1;
    for (size_t i = 0; i + 1 < (*xt)->shape().rank(); ++i) {
        M *= static_cast<int>((*xt)->shape().dim(i));
    }
    int K = static_cast<int>((*xt)->shape().dim((*xt)->shape().rank() - 1));
    int N = static_cast<int>((*wt)->shape().dim(0));

    cpu::sgemm_nt(float_data(*xt), float_data(*wt), float_data_mut(*ot), M, N, K);
    return Status::make_ok();
}

static Status kernel_matmul(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*at, "a"); if (!st.ok()) return st;
    st = check_allocated(*bt, "b"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    int M = static_cast<int>((*at)->shape().dim(0));
    int K = static_cast<int>((*at)->shape().dim(1));
    int N = static_cast<int>((*bt)->shape().dim(1));

    cpu::sgemm(float_data(*at), float_data(*bt), float_data_mut(*ot), M, N, K);
    return Status::make_ok();
}

static Status kernel_rmsnorm(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto gt = get_tensor(node.inputs()[1], ctx, "gamma");
    if (!gt) return gt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x"); if (!st.ok()) return st;
    st = check_allocated(*gt, "gamma"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    int rows = 1;
    for (size_t i = 0; i + 1 < (*xt)->shape().rank(); ++i) {
        rows *= static_cast<int>((*xt)->shape().dim(i));
    }
    int hidden = static_cast<int>((*xt)->shape().dim((*xt)->shape().rank() - 1));

    double eps = 1e-6;
    if (auto attr = node.get_attr("eps")) {
        if (auto* p = std::get_if<double>(&*attr)) eps = *p;
    }

    cpu::rmsnorm(float_data(*xt), float_data(*gt), float_data_mut(*ot),
                 rows, hidden, static_cast<float>(eps));
    return Status::make_ok();
}

static Status kernel_add(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*at, "a"); if (!st.ok()) return st;
    st = check_allocated(*bt, "b"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    auto n = (*ot)->numel();
    if (!n) return n.error();
    cpu::add(float_data(*at), float_data(*bt), float_data_mut(*ot), static_cast<int>(*n));
    return Status::make_ok();
}

static Status kernel_mul(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*at, "a"); if (!st.ok()) return st;
    st = check_allocated(*bt, "b"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    auto n = (*ot)->numel();
    if (!n) return n.error();
    cpu::mul(float_data(*at), float_data(*bt), float_data_mut(*ot), static_cast<int>(*n));
    return Status::make_ok();
}

static Status kernel_silu(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    auto n = (*ot)->numel();
    if (!n) return n.error();
    cpu::silu(float_data(*xt), float_data_mut(*ot), static_cast<int>(*n));
    return Status::make_ok();
}

static Status kernel_swiglu(const Node& node, RuntimeContext& ctx) {
    auto gt = get_tensor(node.inputs()[0], ctx, "gate");
    if (!gt) return gt.error();
    auto ut = get_tensor(node.inputs()[1], ctx, "up");
    if (!ut) return ut.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*gt, "gate"); if (!st.ok()) return st;
    st = check_allocated(*ut, "up"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    auto n = (*ot)->numel();
    if (!n) return n.error();
    cpu::fused_silu_mul(float_data(*gt), float_data(*ut), float_data_mut(*ot),
                        static_cast<int>(*n));
    return Status::make_ok();
}

static Status kernel_rope(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    int seq_len = 1;
    for (size_t i = 0; i + 1 < (*xt)->shape().rank(); ++i) {
        seq_len *= static_cast<int>((*xt)->shape().dim(i));
    }
    int hidden = static_cast<int>((*xt)->shape().dim((*xt)->shape().rank() - 1));

    int64_t head_dim = 64;
    if (auto attr = node.get_attr("head_dim")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) head_dim = *p;
    }

    // x layout: [batch*seq, num_heads * head_dim] treated as [batch*seq, hidden]
    // We apply RoPE per-head by iterating over heads
    int num_heads = hidden / static_cast<int>(head_dim);
    const float* x_data = float_data(*xt);
    float* o_data = float_data_mut(*ot);

    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            const float* x_head = x_data + s * hidden + h * head_dim;
            float* o_head = o_data + s * hidden + h * head_dim;
            cpu::apply_rope(x_head, o_head, 1, static_cast<int>(head_dim), 10000.0f);
        }
    }
    return Status::make_ok();
}

static Status kernel_attention(const Node& node, RuntimeContext& ctx) {
    auto qt = get_tensor(node.inputs()[0], ctx, "q");
    if (!qt) return qt.error();
    auto kt = get_tensor(node.inputs()[1], ctx, "k");
    if (!kt) return kt.error();
    auto vt = get_tensor(node.inputs()[2], ctx, "v");
    if (!vt) return vt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*qt, "q"); if (!st.ok()) return st;
    st = check_allocated(*kt, "k"); if (!st.ok()) return st;
    st = check_allocated(*vt, "v"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    int64_t num_heads = 12, num_kv_heads = 12, head_dim = 64;
    if (auto attr = node.get_attr("num_heads")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) num_heads = *p;
    }
    if (auto attr = node.get_attr("num_kv_heads")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) num_kv_heads = *p;
    }
    if (auto attr = node.get_attr("head_dim")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) head_dim = *p;
    }

    bool causal = true;
    if (auto attr = node.get_attr("causal")) {
        if (auto* p = std::get_if<bool>(&*attr)) causal = *p;
    }

    int nh = static_cast<int>(num_heads);
    int nkv = static_cast<int>(num_kv_heads);
    int hd = static_cast<int>(head_dim);
    int group_size = nh / nkv;

    // Q: [batch, seq, num_heads * head_dim]
    // K: [batch, seq, num_kv_heads * head_dim]
    // V: [batch, seq, num_kv_heads * head_dim]
    int batch = static_cast<int>((*qt)->shape().dim(0));
    int seq_len = static_cast<int>((*qt)->shape().dim(1));
    int q_hidden = static_cast<int>((*qt)->shape().dim(2));
    int kv_hidden = static_cast<int>((*kt)->shape().dim(2));

    const float* q_data = float_data(*qt);
    const float* k_data = float_data(*kt);
    const float* v_data = float_data(*vt);
    float* o_data = float_data_mut(*ot);

    // Buffers: Q in [nh, seq, hd], K/V expanded to [nh, seq, hd] for GQA
    std::vector<float> q_buf(static_cast<size_t>(nh) * seq_len * hd);
    std::vector<float> k_buf(static_cast<size_t>(nh) * seq_len * hd);
    std::vector<float> v_buf(static_cast<size_t>(nh) * seq_len * hd);
    std::vector<float> o_buf(static_cast<size_t>(nh) * seq_len * hd);

    for (int b = 0; b < batch; ++b) {
        // Reinterleave Q: [seq, nh*hd] -> [nh, seq, hd]
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < nh; ++h) {
                for (int d = 0; d < hd; ++d) {
                    q_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                        q_data[b * seq_len * q_hidden + s * q_hidden + h * hd + d];
                }
            }
        }

        // Reinterleave K/V: [seq, nkv*hd] -> [nh, seq, hd] with GQA expansion
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < nh; ++h) {
                int kv_h = h / group_size;
                for (int d = 0; d < hd; ++d) {
                    auto kv_src = b * seq_len * kv_hidden + s * kv_hidden + kv_h * hd + d;
                    k_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] = k_data[kv_src];
                    v_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] = v_data[kv_src];
                }
            }
        }

        cpu::sdpa(q_buf.data(), k_buf.data(), v_buf.data(), o_buf.data(),
                  nh, seq_len, hd, causal);

        // Reinterleave back: [nh, seq, hd] -> [seq, nh*hd]
        for (int h = 0; h < nh; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < hd; ++d) {
                    auto dst_idx = b * seq_len * q_hidden + s * q_hidden + h * hd + d;
                    o_data[dst_idx] = o_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d];
                }
            }
        }
    }
    return Status::make_ok();
}

static Status kernel_qk_norm(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto gt = get_tensor(node.inputs()[1], ctx, "gamma");
    if (!gt) return gt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x"); if (!st.ok()) return st;
    st = check_allocated(*gt, "gamma"); if (!st.ok()) return st;
    st = check_allocated(*ot, "output"); if (!st.ok()) return st;

    int64_t num_heads = 12, head_dim = 64;
    if (auto attr = node.get_attr("num_heads")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) num_heads = *p;
    }
    if (auto attr = node.get_attr("head_dim")) {
        if (auto* p = std::get_if<int64_t>(&*attr)) head_dim = *p;
    }

    double eps = 1e-6;
    if (auto attr = node.get_attr("eps")) {
        if (auto* p = std::get_if<double>(&*attr)) eps = *p;
    }

    // x: [batch, seq, num_heads * head_dim]
    // Apply RMSNorm per-head: treat as [batch*seq*num_heads, head_dim]
    int nh = static_cast<int>(num_heads);
    int hd = static_cast<int>(head_dim);
    int total = 1;
    for (size_t i = 0; i < (*xt)->shape().rank(); ++i)
        total *= static_cast<int>((*xt)->shape().dim(i));
    int rows = total / hd;

    cpu::rmsnorm(float_data(*xt), float_data(*gt), float_data_mut(*ot),
                 rows, hd, static_cast<float>(eps));
    return Status::make_ok();
}

void register_cpu_kernels(KernelRegistry& registry) {
    registry.register_kernel(DeviceType::CPU, OpType::Embedding, kernel_embedding);
    registry.register_kernel(DeviceType::CPU, OpType::Linear, kernel_linear);
    registry.register_kernel(DeviceType::CPU, OpType::MatMul, kernel_matmul);
    registry.register_kernel(DeviceType::CPU, OpType::RMSNorm, kernel_rmsnorm);
    registry.register_kernel(DeviceType::CPU, OpType::Add, kernel_add);
    registry.register_kernel(DeviceType::CPU, OpType::Mul, kernel_mul);
    registry.register_kernel(DeviceType::CPU, OpType::SiLU, kernel_silu);
    registry.register_kernel(DeviceType::CPU, OpType::SwiGLU, kernel_swiglu);
    registry.register_kernel(DeviceType::CPU, OpType::RoPE, kernel_rope);
    registry.register_kernel(DeviceType::CPU, OpType::Attention, kernel_attention);
    registry.register_kernel(DeviceType::CPU, OpType::QKNorm, kernel_qk_norm);
}

} // namespace minillm
