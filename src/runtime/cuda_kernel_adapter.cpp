#include "minillm/runtime/cuda_kernel_adapter.h"

#include <cuda_runtime.h>

#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <variant>

#include "minillm/core/device.h"
#include "minillm/core/dtype.h"
#include "minillm/core/tensor.h"
#include "minillm/graph/node.h"
#include "minillm/graph/op_type.h"
#include "minillm/runtime/cuda_kernels.h"
#include "minillm/runtime/kernel_adapter_common.h"
#include "minillm/runtime/kernel_registry.h"
#include "minillm/runtime/runtime_context.h"

namespace minillm {

namespace {

Status cuda_status(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return Status::make_ok();
    return Status::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

Status check_allocated(const Tensor* t, std::string_view name) {
    if (!t->is_allocated()) {
        return Status::runtime_error(std::string(name) + " tensor not allocated");
    }
    return Status::make_ok();
}

Status check_cuda_device(const Tensor* t, std::string_view name) {
    if (t->device().type != DeviceType::CUDA) {
        return Status::unsupported(
            std::string(name) + " must be a CUDA tensor, got " + t->device().to_string());
    }
    return Status::make_ok();
}

Status check_dtype_float(const Tensor* t, std::string_view name) {
    if (t->dtype() != DType::Float32) {
        return Status::unsupported(
            std::string(name) + " only supports Float32, got " +
            std::string(dtype_name(t->dtype())));
    }
    return Status::make_ok();
}

Status check_cuda_float(const Tensor* t, std::string_view name) {
    auto st = check_allocated(t, name);
    if (!st.ok()) return st;
    st = check_cuda_device(t, name);
    if (!st.ok()) return st;
    return check_dtype_float(t, name);
}

const float* float_data(const Tensor* t) {
    return reinterpret_cast<const float*>(t->data());
}

float* float_data_mut(Tensor* t) {
    return reinterpret_cast<float*>(t->data());
}

const int* int_data(const Tensor* t) {
    return reinterpret_cast<const int*>(t->data());
}

std::expected<Tensor*, Status> get_tensor(ValueId id, RuntimeContext& ctx,
                                          std::string_view role) {
    auto* t = ctx.get(id);
    if (!t) {
        return std::unexpected(Status::runtime_error(
            std::string(role) + " tensor not found for ValueId %" +
            std::to_string(id.value)));
    }
    return t;
}

std::expected<int, Status> checked_int(size_t value, std::string_view role) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::unexpected(Status::unsupported(
            std::string(role) + " is too large for the current CUDA kernel int indexing"));
    }
    return static_cast<int>(value);
}

std::expected<int, Status> checked_dim(int64_t value, std::string_view role) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        return std::unexpected(Status::unsupported(
            std::string(role) + " dimension is outside CUDA kernel int range"));
    }
    return static_cast<int>(value);
}

std::expected<int, Status> numel_int(const Tensor* t, std::string_view role) {
    auto n = t->numel();
    if (!n) return std::unexpected(n.error());
    return checked_int(*n, role);
}

std::expected<int, Status> rows_before_last_dim(const Tensor* t, std::string_view role) {
    if (t->shape().rank() == 0) {
        return std::unexpected(Status::shape_mismatch(
            std::string(role) + " expects rank >= 1"));
    }
    size_t rows = 1;
    for (size_t i = 0; i + 1 < t->shape().rank(); ++i) {
        rows *= static_cast<size_t>(t->shape().dim(i));
    }
    return checked_int(rows, role);
}

std::expected<int, Status> last_dim(const Tensor* t, std::string_view role) {
    if (t->shape().rank() == 0) {
        return std::unexpected(Status::shape_mismatch(
            std::string(role) + " expects rank >= 1"));
    }
    return checked_dim(t->shape().dim(t->shape().rank() - 1), role);
}

Status kernel_embedding(const Node& node, RuntimeContext& ctx) {
    auto ids_t = get_tensor(node.inputs()[0], ctx, "input_ids");
    if (!ids_t) return ids_t.error();
    auto wt = get_tensor(node.inputs()[1], ctx, "weight");
    if (!wt) return wt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*ids_t, "input_ids");
    if (!st.ok()) return st;
    st = check_cuda_device(*ids_t, "input_ids");
    if (!st.ok()) return st;
    if ((*ids_t)->dtype() != DType::Int32) {
        return Status::unsupported(
            "input_ids only supports Int32, got " +
            std::string(dtype_name((*ids_t)->dtype())));
    }
    st = check_cuda_float(*wt, "weight");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    if ((*wt)->shape().rank() != 2) {
        return Status::shape_mismatch("Embedding weight must be rank 2");
    }
    auto seq_len = numel_int(*ids_t, "embedding input_ids");
    if (!seq_len) return seq_len.error();
    auto vocab_size = checked_dim((*wt)->shape().dim(0), "embedding vocab");
    if (!vocab_size) return vocab_size.error();
    auto hidden = checked_dim((*wt)->shape().dim(1), "embedding hidden");
    if (!hidden) return hidden.error();

    return cuda::embedding(float_data(*wt), int_data(*ids_t), float_data_mut(*ot),
                           *seq_len, *vocab_size, *hidden);
}

Status kernel_linear(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto wt = get_tensor(node.inputs()[1], ctx, "weight");
    if (!wt) return wt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*wt, "weight");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;
    if ((*wt)->shape().rank() != 2) {
        return Status::shape_mismatch("Linear weight must be rank 2");
    }

    auto M = rows_before_last_dim(*xt, "linear x");
    if (!M) return M.error();
    auto K = last_dim(*xt, "linear x");
    if (!K) return K.error();
    auto N = checked_dim((*wt)->shape().dim(0), "linear output");
    if (!N) return N.error();
    auto weight_k = checked_dim((*wt)->shape().dim(1), "linear weight input");
    if (!weight_k) return weight_k.error();
    if (*weight_k != *K) {
        return Status::shape_mismatch("Linear x last dimension must match weight input dimension");
    }

    st = cuda::sgemm_nt(float_data(*xt), float_data(*wt), float_data_mut(*ot),
                        *M, *N, *K);
    if (!st.ok()) return st;

    if (node.inputs().size() >= 3) {
        auto bt = get_tensor(node.inputs()[2], ctx, "bias");
        if (!bt) return bt.error();
        st = check_cuda_float(*bt, "bias");
        if (!st.ok()) return st;
        if ((*bt)->shape().rank() != 1 || (*bt)->shape().dim(0) != *N) {
            return Status::shape_mismatch("Linear bias shape must match output features");
        }
        st = cuda::add_bias(float_data_mut(*ot), float_data(*bt), *M, *N);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status kernel_matmul(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*at, "a");
    if (!st.ok()) return st;
    st = check_cuda_float(*bt, "b");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;
    if ((*at)->shape().rank() != 2 || (*bt)->shape().rank() != 2) {
        return Status::shape_mismatch("CUDA MatMul currently expects rank-2 tensors");
    }

    auto M = checked_dim((*at)->shape().dim(0), "matmul M");
    if (!M) return M.error();
    auto K = checked_dim((*at)->shape().dim(1), "matmul K");
    if (!K) return K.error();
    auto b_k = checked_dim((*bt)->shape().dim(0), "matmul B K");
    if (!b_k) return b_k.error();
    auto N = checked_dim((*bt)->shape().dim(1), "matmul N");
    if (!N) return N.error();
    if (*b_k != *K) {
        return Status::shape_mismatch("MatMul inner dimensions differ");
    }

    return cuda::sgemm(float_data(*at), float_data(*bt), float_data_mut(*ot),
                       *M, *N, *K);
}

Status kernel_rmsnorm(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto gt = get_tensor(node.inputs()[1], ctx, "gamma");
    if (!gt) return gt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*gt, "gamma");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto rows = rows_before_last_dim(*xt, "rmsnorm x");
    if (!rows) return rows.error();
    auto hidden = last_dim(*xt, "rmsnorm x");
    if (!hidden) return hidden.error();
    if ((*gt)->shape().rank() != 1 || (*gt)->shape().dim(0) != *hidden) {
        return Status::shape_mismatch("RMSNorm gamma shape must match hidden size");
    }
    double eps = detail::double_attr(node, "eps", 1e-6);
    return cuda::rmsnorm(float_data(*xt), float_data(*gt), float_data_mut(*ot),
                         *rows, *hidden, static_cast<float>(eps));
}

Status kernel_add(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*at, "a");
    if (!st.ok()) return st;
    st = check_cuda_float(*bt, "b");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto n = numel_int(*ot, "add output");
    if (!n) return n.error();
    return cuda::add(float_data(*at), float_data(*bt), float_data_mut(*ot), *n);
}

Status kernel_mul(const Node& node, RuntimeContext& ctx) {
    auto at = get_tensor(node.inputs()[0], ctx, "a");
    if (!at) return at.error();
    auto bt = get_tensor(node.inputs()[1], ctx, "b");
    if (!bt) return bt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*at, "a");
    if (!st.ok()) return st;
    st = check_cuda_float(*bt, "b");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto n = numel_int(*ot, "mul output");
    if (!n) return n.error();
    return cuda::mul(float_data(*at), float_data(*bt), float_data_mut(*ot), *n);
}

Status kernel_silu(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto n = numel_int(*ot, "silu output");
    if (!n) return n.error();
    return cuda::silu(float_data(*xt), float_data_mut(*ot), *n);
}

Status kernel_swiglu(const Node& node, RuntimeContext& ctx) {
    auto gt = get_tensor(node.inputs()[0], ctx, "gate");
    if (!gt) return gt.error();
    auto ut = get_tensor(node.inputs()[1], ctx, "up");
    if (!ut) return ut.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*gt, "gate");
    if (!st.ok()) return st;
    st = check_cuda_float(*ut, "up");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto n = numel_int(*ot, "swiglu output");
    if (!n) return n.error();
    return cuda::fused_silu_mul(float_data(*gt), float_data(*ut), float_data_mut(*ot), *n);
}

Status kernel_softmax(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    int64_t axis = detail::int_attr(node, "axis", -1);
    const auto& shape = (*xt)->shape();
    int64_t rank = static_cast<int64_t>(shape.rank());
    if (rank <= 0) {
        return Status::shape_mismatch("Softmax expects rank >= 1");
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        return Status::out_of_range("Softmax axis out of range");
    }
    if (axis != rank - 1) {
        return Status::unsupported("CUDA Softmax currently supports the last axis only");
    }

    auto cols = checked_dim(shape.dim(static_cast<size_t>(axis)), "softmax cols");
    if (!cols) return cols.error();
    size_t rows_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        rows_size *= static_cast<size_t>(shape.dim(static_cast<size_t>(i)));
    }
    auto rows = checked_int(rows_size, "softmax rows");
    if (!rows) return rows.error();
    return cuda::softmax(float_data(*xt), float_data_mut(*ot), *rows, *cols);
}

Status kernel_reshape(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_allocated(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_device(*xt, "x");
    if (!st.ok()) return st;
    st = check_allocated(*ot, "output");
    if (!st.ok()) return st;
    st = check_cuda_device(*ot, "output");
    if (!st.ok()) return st;
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
    return cuda_status(cudaMemcpy((*ot)->data(), (*xt)->data(), *bytes,
                                  cudaMemcpyDeviceToDevice),
                       "cuda reshape copy failed");
}

Status kernel_transpose(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    int64_t axis0 = detail::int_attr(node, "axis0", -2);
    int64_t axis1 = detail::int_attr(node, "axis1", -1);
    const auto& shape = (*xt)->shape();
    int64_t rank = static_cast<int64_t>(shape.rank());
    if (rank <= 0) {
        return Status::shape_mismatch("Transpose expects rank >= 1");
    }
    if (axis0 < 0) axis0 += rank;
    if (axis1 < 0) axis1 += rank;
    if (axis0 < 0 || axis0 >= rank || axis1 < 0 || axis1 >= rank) {
        return Status::out_of_range("Transpose axes out of range");
    }

    return cuda::transpose(float_data(*xt), float_data_mut(*ot), shape.dims().data(),
                           static_cast<int>(rank), static_cast<int>(axis0),
                           static_cast<int>(axis1));
}

Status kernel_rope(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    auto tokens = rows_before_last_dim(*xt, "rope x");
    if (!tokens) return tokens.error();
    auto hidden = last_dim(*xt, "rope x");
    if (!hidden) return hidden.error();
    int64_t head_dim_attr = detail::int_attr(node, "head_dim", 64);
    auto head_dim = checked_dim(head_dim_attr, "rope head_dim");
    if (!head_dim) return head_dim.error();
    if (*head_dim <= 0 || (*head_dim % 2) != 0) {
        return Status::invalid_argument("RoPE head_dim must be a positive even value");
    }
    int64_t heads_attr = detail::int_attr(node, "num_heads", *hidden / *head_dim);
    auto num_heads = checked_dim(heads_attr, "rope num_heads");
    if (!num_heads) return num_heads.error();
    if (*hidden != *num_heads * *head_dim) {
        return Status::shape_mismatch("RoPE hidden size must match num_heads * head_dim");
    }

    int pos_offset = 0;
    if (auto* cache = ctx.kv_cache(); cache && cache->initialized() &&
        cache->cached_len() > 0 && *tokens == 1) {
        pos_offset = cache->cached_len();
    }
    double base = detail::double_attr(node, "base", 10000.0);
    return cuda::apply_rope(float_data(*xt), float_data_mut(*ot), *tokens,
                            *num_heads, *head_dim, static_cast<float>(base),
                            pos_offset);
}

Status kernel_attention(const Node& node, RuntimeContext& ctx) {
    auto qt = get_tensor(node.inputs()[0], ctx, "q");
    if (!qt) return qt.error();
    auto kt = get_tensor(node.inputs()[1], ctx, "k");
    if (!kt) return kt.error();
    auto vt = get_tensor(node.inputs()[2], ctx, "v");
    if (!vt) return vt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*qt, "q");
    if (!st.ok()) return st;
    st = check_cuda_float(*kt, "k");
    if (!st.ok()) return st;
    st = check_cuda_float(*vt, "v");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;
    if ((*qt)->shape().rank() != 3 || (*kt)->shape().rank() != 3 ||
        (*vt)->shape().rank() != 3) {
        return Status::shape_mismatch("CUDA Attention expects q/k/v rank-3 tensors");
    }

    int64_t num_heads_attr = detail::int_attr(node, "num_heads", 12);
    int64_t num_kv_heads_attr = detail::int_attr(node, "num_kv_heads", num_heads_attr);
    int64_t head_dim_attr = detail::int_attr(node, "head_dim", 64);
    auto num_heads = checked_dim(num_heads_attr, "attention num_heads");
    if (!num_heads) return num_heads.error();
    auto num_kv_heads = checked_dim(num_kv_heads_attr, "attention num_kv_heads");
    if (!num_kv_heads) return num_kv_heads.error();
    auto head_dim = checked_dim(head_dim_attr, "attention head_dim");
    if (!head_dim) return head_dim.error();
    if (*num_heads <= 0 || *num_kv_heads <= 0 || *head_dim <= 0) {
        return Status::invalid_argument(
            "attention requires positive num_heads, num_kv_heads, and head_dim");
    }
    if (*num_heads % *num_kv_heads != 0) {
        return Status::invalid_argument("attention num_heads must be divisible by num_kv_heads");
    }

    auto batch = checked_dim((*qt)->shape().dim(0), "attention batch");
    if (!batch) return batch.error();
    auto q_len = checked_dim((*qt)->shape().dim(1), "attention q_len");
    if (!q_len) return q_len.error();
    auto q_hidden = checked_dim((*qt)->shape().dim(2), "attention q hidden");
    if (!q_hidden) return q_hidden.error();
    auto kv_hidden = checked_dim((*kt)->shape().dim(2), "attention kv hidden");
    if (!kv_hidden) return kv_hidden.error();
    if (*q_hidden != *num_heads * *head_dim) {
        return Status::shape_mismatch(
            "attention q hidden size does not match num_heads * head_dim");
    }
    if (*kv_hidden != *num_kv_heads * *head_dim ||
        (*vt)->shape().dim(2) != *kv_hidden) {
        return Status::shape_mismatch(
            "attention k/v hidden size does not match num_kv_heads * head_dim");
    }
    if ((*kt)->shape().dim(0) != *batch || (*vt)->shape().dim(0) != *batch ||
        (*kt)->shape().dim(1) != *q_len || (*vt)->shape().dim(1) != *q_len) {
        return Status::shape_mismatch("CUDA Attention currently expects q/k/v to share batch and seq");
    }

    bool causal = detail::bool_attr(node, "causal", true);
    KVCache* cache = ctx.kv_cache();
    if (cache && cache->initialized()) {
        if (!cache->is_cuda()) {
            return Status::unsupported("CUDA Attention requires a CUDA KV cache");
        }
        if (*batch != 1) {
            return Status::unsupported("CUDA KV cache attention currently supports batch size 1");
        }
        if (cache->kv_hidden() != *kv_hidden) {
            return Status::shape_mismatch(
                "CUDA KV cache hidden size does not match attention K/V hidden size");
        }

        int64_t layer_idx_attr = detail::int_attr(node, "layer_idx", 0);
        auto layer_idx = checked_dim(layer_idx_attr, "attention layer_idx");
        if (!layer_idx) return layer_idx.error();
        if (*layer_idx < 0 || *layer_idx >= cache->num_layers()) {
            return Status::out_of_range("CUDA KV cache layer index out of range");
        }

        float* cache_k = cache->k_data(*layer_idx);
        float* cache_v = cache->v_data(*layer_idx);
        int cached_len = cache->cached_len();
        if (cached_len < 0 || cached_len > cache->max_seq_len()) {
            return Status::out_of_range("CUDA KV cache cached_len is out of range");
        }

        if (cached_len == 0) {
            if (!cache->can_append(*q_len)) {
                return Status::out_of_range(
                    "CUDA KV cache does not have enough space for prefill");
            }
            const size_t bytes =
                static_cast<size_t>(*q_len) * static_cast<size_t>(*kv_hidden) * sizeof(float);
            auto st = cuda_status(cudaMemcpy(cache_k, float_data(*kt), bytes,
                                             cudaMemcpyDeviceToDevice),
                                  "copy CUDA prefill K into KV cache failed");
            if (!st.ok()) return st;
            st = cuda_status(cudaMemcpy(cache_v, float_data(*vt), bytes,
                                        cudaMemcpyDeviceToDevice),
                             "copy CUDA prefill V into KV cache failed");
            if (!st.ok()) return st;

            return cuda::sdpa(float_data(*qt), float_data(*kt), float_data(*vt),
                              float_data_mut(*ot), *batch, *q_len, *num_heads,
                              *num_kv_heads, *head_dim, causal);
        }

        if (*q_len != 1) {
            return Status::invalid_argument("CUDA KV cache decode requires q_len == 1");
        }
        if (!cache->can_append(1)) {
            return Status::out_of_range("CUDA KV cache does not have enough space for decode");
        }

        const size_t row_offset =
            static_cast<size_t>(cached_len) * static_cast<size_t>(*kv_hidden);
        const size_t bytes = static_cast<size_t>(*kv_hidden) * sizeof(float);
        auto st = cuda_status(cudaMemcpy(cache_k + row_offset, float_data(*kt), bytes,
                                         cudaMemcpyDeviceToDevice),
                              "copy CUDA decode K into KV cache failed");
        if (!st.ok()) return st;
        st = cuda_status(cudaMemcpy(cache_v + row_offset, float_data(*vt), bytes,
                                    cudaMemcpyDeviceToDevice),
                         "copy CUDA decode V into KV cache failed");
        if (!st.ok()) return st;

        return cuda::kv_cache_attention_decode(
            float_data(*qt), cache_k, cache_v, float_data_mut(*ot), cached_len + 1,
            *num_heads, *num_kv_heads, *head_dim);
    }

    return cuda::sdpa(float_data(*qt), float_data(*kt), float_data(*vt),
                      float_data_mut(*ot), *batch, *q_len, *num_heads,
                      *num_kv_heads, *head_dim, causal);
}

Status kernel_qk_norm(const Node& node, RuntimeContext& ctx) {
    auto xt = get_tensor(node.inputs()[0], ctx, "x");
    if (!xt) return xt.error();
    auto gt = get_tensor(node.inputs()[1], ctx, "gamma");
    if (!gt) return gt.error();
    auto ot = get_tensor(node.outputs()[0], ctx, "output");
    if (!ot) return ot.error();

    auto st = check_cuda_float(*xt, "x");
    if (!st.ok()) return st;
    st = check_cuda_float(*gt, "gamma");
    if (!st.ok()) return st;
    st = check_cuda_float(*ot, "output");
    if (!st.ok()) return st;

    int64_t head_dim_attr = detail::int_attr(node, "head_dim", 64);
    auto head_dim = checked_dim(head_dim_attr, "qk_norm head_dim");
    if (!head_dim) return head_dim.error();
    if (*head_dim <= 0) {
        return Status::invalid_argument("QKNorm head_dim must be positive");
    }
    if ((*gt)->shape().rank() != 1 || (*gt)->shape().dim(0) != *head_dim) {
        return Status::shape_mismatch("QKNorm gamma shape must match head_dim");
    }
    auto total = (*xt)->numel();
    if (!total) return total.error();
    if (*total % static_cast<size_t>(*head_dim) != 0) {
        return Status::shape_mismatch("QKNorm tensor size must be divisible by head_dim");
    }
    auto rows = checked_int(*total / static_cast<size_t>(*head_dim), "qk_norm rows");
    if (!rows) return rows.error();
    double eps = detail::double_attr(node, "eps", 1e-6);
    return cuda::rmsnorm(float_data(*xt), float_data(*gt), float_data_mut(*ot),
                         *rows, *head_dim, static_cast<float>(eps));
}

} // namespace

void register_cuda_kernels(KernelRegistry& registry) {
    registry.register_kernel(DeviceType::CUDA, OpType::Embedding, kernel_embedding);
    registry.register_kernel(DeviceType::CUDA, OpType::Linear, kernel_linear);
    registry.register_kernel(DeviceType::CUDA, OpType::MatMul, kernel_matmul);
    registry.register_kernel(DeviceType::CUDA, OpType::RMSNorm, kernel_rmsnorm);
    registry.register_kernel(DeviceType::CUDA, OpType::Add, kernel_add);
    registry.register_kernel(DeviceType::CUDA, OpType::Mul, kernel_mul);
    registry.register_kernel(DeviceType::CUDA, OpType::SiLU, kernel_silu);
    registry.register_kernel(DeviceType::CUDA, OpType::SwiGLU, kernel_swiglu);
    registry.register_kernel(DeviceType::CUDA, OpType::RoPE, kernel_rope);
    registry.register_kernel(DeviceType::CUDA, OpType::Attention, kernel_attention);
    registry.register_kernel(DeviceType::CUDA, OpType::QKNorm, kernel_qk_norm);
    registry.register_kernel(DeviceType::CUDA, OpType::Softmax, kernel_softmax);
    registry.register_kernel(DeviceType::CUDA, OpType::Reshape, kernel_reshape);
    registry.register_kernel(DeviceType::CUDA, OpType::Transpose, kernel_transpose);
}

} // namespace minillm
