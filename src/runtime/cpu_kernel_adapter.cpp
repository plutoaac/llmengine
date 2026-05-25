#include "minillm/runtime/cpu_kernel_adapter.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "minillm/core/dtype.h"
#include "minillm/core/shape.h"
#include "minillm/core/tensor.h"
#include "minillm/graph/node.h"
#include "minillm/graph/op_type.h"
#include "minillm/runtime/cpu_kernels.h"
#include "minillm/runtime/kernel_adapter_common.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {
using namespace detail;

static Status kernel_embedding(const Node& node, RuntimeContext& ctx) {
    auto ids_t = get_tensor(node.inputs()[0], ctx, "input_ids");
    if (!ids_t) return ids_t.error();
    auto wt = get_tensor(node.inputs()[1], ctx, "weight");
    if (!wt) return wt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*ids_t, "input_ids"));
    TRY(check_allocated(*wt, "weight"));
    TRY(check_allocated(*ot, "output"));
    TRY(check_dtype_float(*wt, "weight"));
    TRY(check_dtype_float(*ot, "output"));
    if ((*ids_t)->dtype() != DType::Int32) {
        return Status::unsupported(
            "input_ids only supports Int32, got " +
            std::string(dtype_name((*ids_t)->dtype())));
    }

    auto ids_numel = (*ids_t)->numel();
    if (!ids_numel) return ids_numel.error();
    int seq_len = static_cast<int>(*ids_numel);
    int hidden = static_cast<int>((*wt)->shape().dim(1));
    int vocab_size = static_cast<int>((*wt)->shape().dim(0));
    const int* ids = int_data(*ids_t);
    for (int i = 0; i < seq_len; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab_size) {
            return Status::out_of_range(
                "input_ids contains token id out of embedding vocabulary range");
        }
    }
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

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*wt, "weight"));
    TRY(check_allocated(*ot, "output"));
    TRY(check_dtype_float(*xt, "x"));
    TRY(check_dtype_float(*wt, "weight"));
    TRY(check_dtype_float(*ot, "output"));

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
    if (node.inputs().size() >= 3) {
        auto bt = get_tensor(node.inputs()[2], ctx, "bias");
        if (!bt) return bt.error();
        TRY(check_allocated(*bt, "bias"));
        TRY(check_dtype_float(*bt, "bias"));
        if ((*bt)->shape().rank() != 1 || (*bt)->shape().dim(0) != N) {
            return Status::shape_mismatch("Linear bias shape must match output features");
        }

        const float* bias = float_data(*bt);
        float* out = float_data_mut(*ot);
        for (int m = 0; m < M; ++m) {
            cpu::add(out + m * N, bias, out + m * N, N);
        }
    }
    return Status::make_ok();
}

static Status kernel_matmul(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*at, "a"));
    TRY(check_allocated(*bt, "b"));
    TRY(check_allocated(*ot, "output"));

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

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*gt, "gamma"));
    TRY(check_allocated(*ot, "output"));

    int rows = 1;
    for (size_t i = 0; i + 1 < (*xt)->shape().rank(); ++i) {
        rows *= static_cast<int>((*xt)->shape().dim(i));
    }
    int hidden = static_cast<int>((*xt)->shape().dim((*xt)->shape().rank() - 1));

    double eps = detail::double_attr(node, "eps", 1e-6);

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

    TRY(check_allocated(*at, "a"));
    TRY(check_allocated(*bt, "b"));
    TRY(check_allocated(*ot, "output"));

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

    TRY(check_allocated(*at, "a"));
    TRY(check_allocated(*bt, "b"));
    TRY(check_allocated(*ot, "output"));

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

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*ot, "output"));

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

    TRY(check_allocated(*gt, "gate"));
    TRY(check_allocated(*ut, "up"));
    TRY(check_allocated(*ot, "output"));

    auto n = (*ot)->numel();
    if (!n) return n.error();
    cpu::fused_silu_mul(float_data(*gt), float_data(*ut), float_data_mut(*ot),
                        static_cast<int>(*n));
    return Status::make_ok();
}

static Status kernel_softmax(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*ot, "output"));
    TRY(check_dtype_float(*xt, "x"));
    TRY(check_dtype_float(*ot, "output"));

    int64_t axis = detail::int_attr(node, "axis", -1);

    const auto& shape = (*xt)->shape();
    const int64_t rank = static_cast<int64_t>(shape.rank());
    if (rank <= 0) {
        return Status::shape_mismatch("Softmax expects rank >= 1");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        return Status::out_of_range("Softmax axis out of range");
    }
    if (axis != rank - 1) {
        return Status::unsupported("CPU Softmax currently supports the last axis only");
    }

    int cols = static_cast<int>(shape.dim(static_cast<size_t>(axis)));
    int rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        rows *= static_cast<int>(shape.dim(static_cast<size_t>(i)));
    }
    cpu::softmax(float_data(*xt), float_data_mut(*ot), rows, cols);
    return Status::make_ok();
}

static Status kernel_reshape(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*ot, "output"));
    if ((*xt)->dtype() != (*ot)->dtype()) {
        return Status::unsupported("Reshape requires matching input/output dtypes");
    }

    auto in_numel = (*xt)->numel();
    if (!in_numel) return in_numel.error();
    auto out_numel = (*ot)->numel();
    if (!out_numel) return out_numel.error();
    if (*in_numel != *out_numel) {
        return Status::shape_mismatch("Reshape input/output element counts differ");
    }

    auto bytes = (*xt)->nbytes();
    if (!bytes) return bytes.error();
    auto out_bytes = (*ot)->nbytes();
    if (!out_bytes) return out_bytes.error();
    if (*bytes != *out_bytes) {
        return Status::shape_mismatch("Reshape input/output byte sizes differ");
    }
    std::memcpy((*ot)->data(), (*xt)->data(), *bytes);
    return Status::make_ok();
}

static Status kernel_transpose(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*ot, "output"));
    TRY(check_dtype_float(*xt, "x"));
    TRY(check_dtype_float(*ot, "output"));

    int64_t axis0 = detail::int_attr(node, "axis0", -2);
    int64_t axis1 = detail::int_attr(node, "axis1", -1);

    const auto& shape = (*xt)->shape();
    const int64_t rank = static_cast<int64_t>(shape.rank());
    if (rank <= 0) {
        return Status::shape_mismatch("Transpose expects rank >= 1");
    }
    if (axis0 < 0) axis0 += rank;
    if (axis1 < 0) axis1 += rank;
    if (axis0 < 0 || axis0 >= rank || axis1 < 0 || axis1 >= rank) {
        return Status::out_of_range("Transpose axes out of range");
    }

    auto in_numel = (*xt)->numel();
    if (!in_numel) return in_numel.error();
    auto out_numel = (*ot)->numel();
    if (!out_numel) return out_numel.error();
    if (*in_numel != *out_numel) {
        return Status::shape_mismatch("Transpose input/output element counts differ");
    }

    cpu::transpose(float_data(*xt), float_data_mut(*ot), shape.dims().data(),
                   static_cast<int>(rank), static_cast<int>(axis0), static_cast<int>(axis1));
    return Status::make_ok();
}

static Status kernel_rope(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*ot, "output"));

    int seq_len = 1;
    for (size_t i = 0; i + 1 < (*xt)->shape().rank(); ++i) {
        seq_len *= static_cast<int>((*xt)->shape().dim(i));
    }
    int hidden = static_cast<int>((*xt)->shape().dim((*xt)->shape().rank() - 1));

    int64_t head_dim = detail::int_attr(node, "head_dim", 64);

    int pos_offset = 0;
    if (auto* pc = ctx.paged_kv_cache(); pc && pc->initialized()) {
        int wp = ctx.paged_kv_write_pos();
        pos_offset = (wp >= 0) ? wp : pc->sequence_length(ctx.paged_sequence_id());
    } else if (KVCache* cache = ctx.kv_cache();
               cache && cache->initialized() && cache->cached_len() > 0 && seq_len == 1) {
        pos_offset = cache->cached_len();
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
            float rope_base = static_cast<float>(detail::double_attr(node, "rope_base", 10000.0));
            cpu::apply_rope(x_head, o_head, 1, static_cast<int>(head_dim), rope_base, pos_offset + s);
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

    TRY(check_allocated(*qt, "q"));
    TRY(check_allocated(*kt, "k"));
    TRY(check_allocated(*vt, "v"));
    TRY(check_allocated(*ot, "output"));

    int64_t num_heads = detail::int_attr(node, "num_heads", 12);
    int64_t num_kv_heads = detail::int_attr(node, "num_kv_heads", 12);
    int64_t head_dim = detail::int_attr(node, "head_dim", 64);
    int64_t layer_idx = detail::int_attr(node, "layer_idx", 0);
    bool causal = detail::bool_attr(node, "causal", true);

    int nh = static_cast<int>(num_heads);
    int nkv = static_cast<int>(num_kv_heads);
    int hd = static_cast<int>(head_dim);
    if (nh <= 0 || nkv <= 0 || hd <= 0) {
        return Status::invalid_argument(
            "attention requires positive num_heads, num_kv_heads, and head_dim");
    }
    if (nh % nkv != 0) {
        return Status::invalid_argument(
            "attention num_heads must be divisible by num_kv_heads");
    }
    int group_size = nh / nkv;

    // Q: [batch, seq, num_heads * head_dim]
    // K: [batch, seq, num_kv_heads * head_dim]
    // V: [batch, seq, num_kv_heads * head_dim]
    int batch = static_cast<int>((*qt)->shape().dim(0));
    int seq_len = static_cast<int>((*qt)->shape().dim(1));
    int q_hidden = static_cast<int>((*qt)->shape().dim(2));
    int kv_hidden_size = static_cast<int>((*kt)->shape().dim(2));
    if (q_hidden != nh * hd) {
        return Status::shape_mismatch(
            "attention q hidden size does not match num_heads * head_dim");
    }
    if (kv_hidden_size != nkv * hd) {
        return Status::shape_mismatch(
            "attention k/v hidden size does not match num_kv_heads * head_dim");
    }

    const float* q_data = float_data(*qt);
    const float* k_data = float_data(*kt);
    const float* v_data = float_data(*vt);
    float* o_data = float_data_mut(*ot);

    // =================== PAGED ATTENTION PATH ===================
    // When a paged KV cache is attached to the context, the attention kernel
    // writes newly projected K/V directly into paged storage and reads all
    // previous K/V from it, bypassing the contiguous KVCache entirely. Check
    // this first because a paged decode context may have no contiguous KVCache.
    if (auto* paged = ctx.paged_kv_cache(); paged && paged->initialized()) {
        const int li = static_cast<int>(layer_idx);
        const int seq_id = ctx.paged_sequence_id();
        int write_pos = ctx.paged_kv_write_pos();
        if (write_pos < 0) {
            write_pos = paged->sequence_length(seq_id);
            ctx.set_paged_kv_write_pos(write_pos);
        }

        // All layers in one forward pass must write at the same logical token
        // position. PagedKVCache::write_tokens updates sequence_length, so using
        // sequence_length again on later layers would incorrectly advance once
        // per layer instead of once per generated token.
        for (int s = 0; s < seq_len; ++s) {
            const float* k_row = k_data + s * kv_hidden_size;
            const float* v_row = v_data + s * kv_hidden_size;
            auto st = paged->write_tokens(seq_id, li, write_pos + s,
                                          k_row, v_row, 1);
            if (!st.ok()) return st;
        }

        // Reinterleave Q as [nh, seq_len, hd]
        std::vector<float> q_buf(static_cast<size_t>(nh) * seq_len * hd);
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < nh; ++h) {
                for (int d = 0; d < hd; ++d) {
                    q_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                        q_data[s * q_hidden + h * hd + d];
                }
            }
        }

        std::vector<float> o_buf(static_cast<size_t>(nh) * seq_len * hd);

        if (seq_len == 1) {
            // Decode: use optimized single-token attention
            auto st = paged_attention_decode(*paged, seq_id, li,
                                             q_buf.data(), o_buf.data(), nh, hd);
            if (!st.ok()) return st;
        } else {
            // Prefill/chunked decode: query qs may only attend through its own
            // absolute position, not future tokens from the same chunk.
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            for (int h = 0; h < nh; ++h) {
                const int kv_h_idx = h / group_size;
                const float* q_vec = q_buf.data() + static_cast<size_t>(h) * seq_len * hd;
                float* o_vec = o_buf.data() + static_cast<size_t>(h) * seq_len * hd;

                for (int qs = 0; qs < seq_len; ++qs) {
                    const float* q_row = q_vec + qs * hd;
                    const int kv_limit = write_pos + qs + 1;
                    std::vector<float> scores(static_cast<size_t>(kv_limit));
                    float max_score = -std::numeric_limits<float>::max();

                    for (int pos = 0; pos < kv_limit; ++pos) {
                        auto k_ptr = paged->key_ptr(seq_id, li, pos, kv_h_idx);
                        if (!k_ptr) return k_ptr.error();
                        float dot = 0.0f;
                        for (int d = 0; d < hd; ++d) {
                            dot += q_row[d] * (*k_ptr)[d];
                        }
                        float s_val = dot * scale;
                        scores[static_cast<size_t>(pos)] = s_val;
                        max_score = std::max(max_score, s_val);
                    }

                    float sum = 0.0f;
                    for (int pos = 0; pos < kv_limit; ++pos) {
                        float p = std::exp(scores[static_cast<size_t>(pos)] - max_score);
                        scores[static_cast<size_t>(pos)] = p;
                        sum += p;
                    }
                    if (sum <= 0.0f) {
                        return Status::runtime_error("paged attention softmax sum is non-positive");
                    }

                    float* out_row = o_vec + qs * hd;
                    std::fill(out_row, out_row + hd, 0.0f);
                    for (int pos = 0; pos < kv_limit; ++pos) {
                        auto v_ptr = paged->value_ptr(seq_id, li, pos, kv_h_idx);
                        if (!v_ptr) return v_ptr.error();
                        const float weight = scores[static_cast<size_t>(pos)] / sum;
                        for (int d = 0; d < hd; ++d) {
                            out_row[d] += weight * (*v_ptr)[d];
                        }
                    }
                }
            }
        }

        // Write output
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < nh; ++h) {
                for (int d = 0; d < hd; ++d) {
                    o_data[s * q_hidden + h * hd + d] =
                        o_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d];
                }
            }
        }
        return Status::make_ok();
    }

    // =================== NO CACHE PATH (backward compat) ===================
    KVCache* cache = ctx.kv_cache();
    if (!cache || !cache->initialized()) {
        std::vector<float> q_buf(static_cast<size_t>(nh) * seq_len * hd);
        std::vector<float> k_buf(static_cast<size_t>(nh) * seq_len * hd);
        std::vector<float> v_buf(static_cast<size_t>(nh) * seq_len * hd);
        std::vector<float> o_buf(static_cast<size_t>(nh) * seq_len * hd);

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < nh; ++h) {
                    for (int d = 0; d < hd; ++d) {
                        q_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                            q_data[b * seq_len * q_hidden + s * q_hidden + h * hd + d];
                    }
                }
            }
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < nh; ++h) {
                    int kv_h = h / group_size;
                    for (int d = 0; d < hd; ++d) {
                        auto kv_src = b * seq_len * kv_hidden_size + s * kv_hidden_size + kv_h * hd + d;
                        k_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] = k_data[kv_src];
                        v_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] = v_data[kv_src];
                    }
                }
            }

            cpu::flash_sdpa(q_buf.data(), k_buf.data(), v_buf.data(), o_buf.data(),
                      nh, seq_len, seq_len, hd, causal);

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

    // =================== CACHE PATH ===================
    int li = static_cast<int>(layer_idx);
    if (batch != 1) {
        return Status::unsupported("KV cache attention currently supports batch size 1");
    }
    if (seq_len <= 0) {
        return Status::invalid_argument("KV cache attention requires seq_len > 0");
    }
    if (li < 0 || li >= cache->num_layers()) {
        return Status::out_of_range("KV cache layer index out of range");
    }
    float* cache_k = cache->k_data(li);
    float* cache_v = cache->v_data(li);
    int cached_len = cache->cached_len();
    if (cached_len < 0 || cached_len > cache->max_seq_len()) {
        return Status::out_of_range("KV cache cached_len is out of range");
    }
    int kv_h = nkv * hd;  // kv_hidden per position

    if (cached_len == 0) {
        if (!cache->can_append(seq_len)) {
            return Status::out_of_range("KV cache does not have enough space for prefill");
        }

        // Copy K/V rows into cache: [seq, nkv*hd] -> cache[0..seq_len-1]
        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                const float* k_row = k_data + b * seq_len * kv_hidden_size + s * kv_hidden_size;
                const float* v_row = v_data + b * seq_len * kv_hidden_size + s * kv_hidden_size;
                std::memcpy(cache_k + s * kv_h, k_row, kv_h * sizeof(float));
                std::memcpy(cache_v + s * kv_h, v_row, kv_h * sizeof(float));
            }
        }

        // Reinterleave Q as [nh, seq, hd]
        std::vector<float> q_buf(static_cast<size_t>(nh) * seq_len * hd);
        // Reinterleave cached K/V into [nh, seq, hd] with GQA expansion
        std::vector<float> k_buf(static_cast<size_t>(nh) * seq_len * hd);
        std::vector<float> v_buf(static_cast<size_t>(nh) * seq_len * hd);
        std::vector<float> o_buf(static_cast<size_t>(nh) * seq_len * hd);

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < nh; ++h) {
                    for (int d = 0; d < hd; ++d) {
                        q_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                            q_data[b * seq_len * q_hidden + s * q_hidden + h * hd + d];
                    }
                }
            }

            // Reinterleave from cache (which has [seq, nkv*hd] layout)
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < nh; ++h) {
                    int kv_hi = h / group_size;
                    for (int d = 0; d < hd; ++d) {
                        k_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                            cache_k[s * kv_h + kv_hi * hd + d];
                        v_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d] =
                            cache_v[s * kv_h + kv_hi * hd + d];
                    }
                }
            }

            cpu::flash_sdpa(q_buf.data(), k_buf.data(), v_buf.data(), o_buf.data(),
                      nh, seq_len, seq_len, hd, causal);

            for (int h = 0; h < nh; ++h) {
                for (int s = 0; s < seq_len; ++s) {
                    for (int d = 0; d < hd; ++d) {
                        auto dst_idx = b * seq_len * q_hidden + s * q_hidden + h * hd + d;
                        o_data[dst_idx] = o_buf[static_cast<size_t>(h) * seq_len * hd + s * hd + d];
                    }
                }
            }
        }

        // Only the last layer should advance the cache
        // We rely on the caller to set cached_len after the full forward pass
        return Status::make_ok();
    }

    // ---- DECODE: cached_len > 0, seq_len should be 1 ----
    {
        if (seq_len != 1) {
            return Status::invalid_argument("KV cache decode requires seq_len == 1");
        }
        if (!cache->can_append(1)) {
            return Status::out_of_range("KV cache does not have enough space for decode");
        }

        for (int b = 0; b < batch; ++b) {
            const float* k_row = k_data + b * seq_len * kv_hidden_size;
            const float* v_row = v_data + b * seq_len * kv_hidden_size;
            std::memcpy(cache_k + cached_len * kv_h, k_row, kv_h * sizeof(float));
            std::memcpy(cache_v + cached_len * kv_h, v_row, kv_h * sizeof(float));
        }

        int new_kv_len = cached_len + 1;

        // Reinterleave Q as [nh, 1, hd]
        std::vector<float> q_buf(static_cast<size_t>(nh) * hd);

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < nh; ++h) {
                for (int d = 0; d < hd; ++d) {
                    q_buf[static_cast<size_t>(h) * hd + d] =
                        q_data[b * seq_len * q_hidden + h * hd + d];
                }
            }

            // Use sdpa_decode which handles GQA directly from cache layout
            std::vector<float> o_buf(static_cast<size_t>(nh) * hd);
            cpu::flash_sdpa_decode(q_buf.data(), cache_k, cache_v, o_buf.data(),
                             nh, nkv, hd, new_kv_len);

            // Copy output: [nh*hd] -> [1, nh*hd]
            for (int i = 0; i < nh * hd; ++i) {
                o_data[i] = o_buf[i];
            }
        }

        // Caller advances cached_len after all layers complete
        return Status::make_ok();
    }
}

static Status kernel_qk_norm(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto gt = get_tensor(node.inputs()[1], ctx, "gamma");
    if (!gt) return gt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    TRY(check_allocated(*xt, "x"));
    TRY(check_allocated(*gt, "gamma"));
    TRY(check_allocated(*ot, "output"));

    int64_t num_heads = detail::int_attr(node, "num_heads", 12);
    int64_t head_dim = detail::int_attr(node, "head_dim", 64);

    double eps = detail::double_attr(node, "eps", 1e-6);

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
    registry.register_kernel(DeviceType::CPU, OpType::Softmax, kernel_softmax);
    registry.register_kernel(DeviceType::CPU, OpType::Reshape, kernel_reshape);
    registry.register_kernel(DeviceType::CPU, OpType::Transpose, kernel_transpose);
    registry.register_kernel(DeviceType::CPU, OpType::RoPE, kernel_rope);
    registry.register_kernel(DeviceType::CPU, OpType::Attention, kernel_attention);
    registry.register_kernel(DeviceType::CPU, OpType::QKNorm, kernel_qk_norm);
}

} // namespace minillm
