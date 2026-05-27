#include "minillm/runtime/kernels/cuda_kernels.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <initializer_list>
#include <utility>
#include <limits>
#include <string>

namespace minillm::cuda {

namespace {

#define RETURN_IF_ERROR(expr) do { Status _s = (expr); if (!_s.ok()) return _s; } while (0)

Status cuda_status(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return Status::make_ok();
    return Status::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

Status launch_status(const char* what) {
    return cuda_status(cudaGetLastError(), what);
}

Status checked_mul_size(size_t a, size_t b, size_t& out, const char* name) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return Status::invalid_argument(std::string("cuda ") + name + " size overflow");
    }
    out = a * b;
    return Status::make_ok();
}

Status checked_mul3_size(size_t a, size_t b, size_t c, size_t& out, const char* name) {
    size_t ab = 0;
    RETURN_IF_ERROR(checked_mul_size(a, b, ab, name));
    return checked_mul_size(ab, c, out, name);
}

Status require_int_indexable(size_t count, const char* name) {
    if (count <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::make_ok();
    }
    return Status::unsupported(std::string("cuda ") + name + " exceeds int-indexed kernel limit");
}

Status require_grid_x(size_t blocks, const char* name) {
    if (blocks == 0) return Status::make_ok();
    if (blocks <= static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
        return Status::make_ok();
    }
    return Status::unsupported(std::string("cuda ") + name + " grid.x is too large");
}

// --- Reusable validation helpers ---

Status require_all_non_null_impl(const char* ctx) {
    (void)ctx;
    return Status::make_ok();
}

template <typename... Rest>
Status require_all_non_null_impl(const char* ctx, const void* ptr, const char* name,
                                 Rest... rest) {
    if (ptr == nullptr) {
        return Status::invalid_argument(
            std::string("cuda ") + ctx + ": " + name + " pointer must not be null");
    }
    return require_all_non_null_impl(ctx, rest...);
}

template <typename... Pairs>
Status require_all_non_null(const char* ctx, Pairs... pairs) {
    return require_all_non_null_impl(ctx, pairs...);
}

Status validate_positive_dims(const char* ctx, std::initializer_list<std::pair<int, const char*>> dims) {
    for (auto& d : dims) {
        if (d.first <= 0) {
            return Status::invalid_argument(
                std::string("cuda ") + ctx + ": " + d.second + " must be positive");
        }
    }
    return Status::make_ok();
}

Status validate_gemm_sizes(int M, int N, int K, const char* ctx) {
    size_t elems = 0;
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(M), static_cast<size_t>(K), elems, ctx));
    RETURN_IF_ERROR(require_int_indexable(elems, ctx));
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(N), static_cast<size_t>(K), elems, ctx));
    RETURN_IF_ERROR(require_int_indexable(elems, ctx));
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(M), static_cast<size_t>(N), elems, ctx));
    return require_int_indexable(elems, ctx);
}

template <typename Fn>
Status launch_1d_grid(size_t n, int block_size, const char* ctx, Fn launcher) {
    RETURN_IF_ERROR(require_int_indexable(n, ctx));
    size_t grid_x = (n + block_size - 1) / block_size;
    RETURN_IF_ERROR(require_grid_x(grid_x, ctx));
    return launcher(static_cast<unsigned>(grid_x));
}

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

__inline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return v;
}

template <int BLOCK>
__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_partials[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_partials[warp_id] = v;
    __syncthreads();

    float block_v = (threadIdx.x < (BLOCK / 32)) ? warp_partials[lane] : 0.0f;
    if (warp_id == 0) block_v = warp_reduce_sum(block_v);
    if (threadIdx.x == 0) warp_partials[0] = block_v;
    __syncthreads();
    float result = warp_partials[0];
    __syncthreads();
    return result;
}

template <int BLOCK>
__inline__ __device__ float block_reduce_max(float v) {
    __shared__ float warp_partials[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) warp_partials[warp_id] = v;
    __syncthreads();

    float block_v = (threadIdx.x < (BLOCK / 32)) ? warp_partials[lane] : -FLT_MAX;
    if (warp_id == 0) block_v = warp_reduce_max(block_v);
    if (threadIdx.x == 0) warp_partials[0] = block_v;
    __syncthreads();
    float result = warp_partials[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float weight_to_f32(float v) {
    return v;
}

__device__ __forceinline__ float weight_to_f32(uint16_t bf16_bits) {
    return __uint_as_float(static_cast<unsigned int>(bf16_bits) << 16);
}

__global__ void add_kernel(const float* a, const float* b, float* y, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

template <typename W>
__global__ void add_bias_kernel(float* y, const W* bias, int cols, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += weight_to_f32(bias[i % static_cast<size_t>(cols)]);
}

__global__ void mul_kernel(const float* a, const float* b, float* y, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

__global__ void silu_kernel(const float* x, float* y, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(x[i]);
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* y, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(gate[i]) * up[i];
}

template <typename W>
__global__ void sgemm_kernel(const float* A, const W* B, float* C,
                             int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        int a_col = tile_k + tx;
        int b_row = tile_k + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N)
            ? weight_to_f32(B[static_cast<size_t>(b_row) * N + col])
            : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

template <typename W>
__global__ void sgemm_nt_kernel(const float* A, const W* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(row) * K + k] *
               weight_to_f32(B[static_cast<size_t>(col) * K + k]);
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

template <typename W, int BLOCK>
__global__ void rmsnorm_kernel(const float* x, const W* gamma, float* y,
                               int rows, int hidden, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + static_cast<size_t>(row) * hidden;
    float* y_row = y + static_cast<size_t>(row) * hidden;

    float local_sumsq = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        float v = x_row[h];
        local_sumsq += v * v;
    }
    float sumsq = block_reduce_sum<BLOCK>(local_sumsq);
    float inv_rms = rsqrtf(sumsq / hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        y_row[h] = x_row[h] * inv_rms * weight_to_f32(gamma[h]);
    }
}

template <typename W, int BLOCK>
__global__ void embedding_kernel(const W* weight, const int* ids, float* out,
                                 int seq_len, int vocab_size, int hidden) {
    int s = blockIdx.x;
    if (s >= seq_len) return;
    int id = ids[s];
    float* dst = out + static_cast<size_t>(s) * hidden;
    if (id < 0 || id >= vocab_size) {
        for (int h = threadIdx.x; h < hidden; h += BLOCK) {
            dst[h] = 0.0f;
        }
        return;
    }
    const W* src = weight + static_cast<size_t>(id) * hidden;
    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        dst[h] = weight_to_f32(src[h]);
    }
}


__global__ void rope_kernel(const float* x, float* y, int tokens, int num_heads,
                            int head_dim, float base, int pos_offset, size_t total) {
    size_t half = static_cast<size_t>(head_dim / 2);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t d = idx % half;
    size_t h = (idx / half) % static_cast<size_t>(num_heads);
    size_t t = idx / (half * static_cast<size_t>(num_heads));
    size_t hidden = static_cast<size_t>(num_heads) * head_dim;
    size_t base_idx = t * hidden + h * static_cast<size_t>(head_dim);
    int pos = pos_offset + static_cast<int>(t);

    float theta = powf(base, -2.0f * static_cast<float>(d) / head_dim) * pos;
    float c = cosf(theta);
    float s = sinf(theta);
    float xe = x[base_idx + d];
    float xo = x[base_idx + d + half];
    y[base_idx + d] = xe * c - xo * s;
    y[base_idx + d + half] = xe * s + xo * c;
}

template <int BLOCK>
__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        local_max = fmaxf(local_max, x_row[c]);
    }
    float max_val = block_reduce_max<BLOCK>(local_max);

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float e = expf(x_row[c] - max_val);
        y_row[c] = e;
        local_sum += e;
    }
    float sum = block_reduce_sum<BLOCK>(local_sum);

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        y_row[c] /= sum;
    }
}

__global__ void transpose_kernel(const float* x, float* y, const int64_t* dims,
                                 int rank, int axis0, int axis1, size_t total) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    size_t rem = linear;
    size_t src = 0;
    size_t in_stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        size_t out_dim = static_cast<size_t>(dims[i == axis0 ? axis1 : (i == axis1 ? axis0 : i)]);
        size_t out_idx = rem % out_dim;
        rem /= out_dim;
        int in_axis = (i == axis0) ? axis1 : (i == axis1 ? axis0 : i);

        size_t stride = 1;
        for (int j = rank - 1; j > in_axis; --j) {
            stride *= static_cast<size_t>(dims[j]);
        }
        src += out_idx * stride;
        in_stride *= static_cast<size_t>(dims[i]);
    }
    (void)in_stride;
    y[linear] = x[src];
}

template <int BLOCK>
__global__ void sdpa_kernel(const float* q, const float* k, const float* v, float* out,
                            int batch, int q_len, int num_heads, int num_kv_heads,
                            int head_dim, bool causal) {
    int q_pos = blockIdx.x;
    int b = blockIdx.y;
    int qh = blockIdx.z;
    int tid = threadIdx.x;

    int group = num_heads / num_kv_heads;
    int kvh = qh / group;
    size_t q_hidden = static_cast<size_t>(num_heads) * head_dim;
    size_t kv_hidden = static_cast<size_t>(num_kv_heads) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + static_cast<size_t>(b) * q_len * q_hidden +
                         static_cast<size_t>(q_pos) * q_hidden +
                         static_cast<size_t>(qh) * head_dim;
    float* out_vec = out + static_cast<size_t>(b) * q_len * q_hidden +
                     static_cast<size_t>(q_pos) * q_hidden +
                     static_cast<size_t>(qh) * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int key_pos = 0; key_pos < q_len; ++key_pos) {
        if (causal && key_pos > q_pos) continue;
        const float* k_vec = k + static_cast<size_t>(b) * q_len * kv_hidden +
                             static_cast<size_t>(key_pos) * kv_hidden +
                             static_cast<size_t>(kvh) * head_dim;
        const float* v_vec = v + static_cast<size_t>(b) * q_len * kv_hidden +
                             static_cast<size_t>(key_pos) * kv_hidden +
                             static_cast<size_t>(kvh) * head_dim;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_vec[d] * k_vec[d];
        }
        float score = block_reduce_sum<BLOCK>(partial) * scale;

        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        if (tid < head_dim) {
            acc = acc * alpha + beta * v_vec[tid];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (tid < head_dim) {
        out_vec[tid] = acc / l;
    }
}

template <int BLOCK>
__global__ void paged_attention_decode_kernel(
    const float* q, const float* k_cache, const float* v_cache,
    const int* block_table, float* out, int seq_len, int num_heads,
    int num_kv_heads, int head_dim, int block_size) {
    int qh = blockIdx.x;
    int tid = threadIdx.x;

    int group = num_heads / num_kv_heads;
    int kvh = qh / group;
    size_t kv_hidden = static_cast<size_t>(num_kv_heads) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + static_cast<size_t>(qh) * head_dim;
    float* out_vec = out + static_cast<size_t>(qh) * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        int physical_block = block_table[pos / block_size];
        if (physical_block < 0) continue;
        int block_offset = pos % block_size;
        size_t base = (static_cast<size_t>(physical_block) * block_size + block_offset) *
                      kv_hidden + static_cast<size_t>(kvh) * head_dim;
        const float* k_vec = k_cache + base;
        const float* v_vec = v_cache + base;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_vec[d] * k_vec[d];
        }
        float score = block_reduce_sum<BLOCK>(partial) * scale;

        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        if (tid < head_dim) {
            acc = acc * alpha + beta * v_vec[tid];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (tid < head_dim) {
        out_vec[tid] = acc / l;
    }
}

template <int BLOCK>
__global__ void paged_attention_decode_batch_kernel(
    const float* q, const float* k_cache, const float* v_cache,
    const int* block_tables, const int* sequence_lengths, float* out,
    int batch_size, int max_blocks_per_sequence, int num_heads,
    int num_kv_heads, int head_dim, int block_size) {
    int qh = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    if (b >= batch_size) return;

    int seq_len = sequence_lengths[b];
    if (seq_len <= 0) return;
    int max_seq_len = max_blocks_per_sequence * block_size;
    if (seq_len > max_seq_len) seq_len = max_seq_len;

    int group = num_heads / num_kv_heads;
    int kvh = qh / group;
    size_t kv_hidden = static_cast<size_t>(num_kv_heads) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int* block_table = block_tables + static_cast<size_t>(b) * max_blocks_per_sequence;

    const float* q_vec = q + (static_cast<size_t>(b) * num_heads + qh) * head_dim;
    float* out_vec = out + (static_cast<size_t>(b) * num_heads + qh) * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        int physical_block = block_table[pos / block_size];
        if (physical_block < 0) continue;
        int block_offset = pos % block_size;
        size_t base = (static_cast<size_t>(physical_block) * block_size + block_offset) *
                      kv_hidden + static_cast<size_t>(kvh) * head_dim;
        const float* k_vec = k_cache + base;
        const float* v_vec = v_cache + base;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_vec[d] * k_vec[d];
        }
        float score = block_reduce_sum<BLOCK>(partial) * scale;

        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        if (tid < head_dim) {
            acc = acc * alpha + beta * v_vec[tid];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (tid < head_dim) {
        out_vec[tid] = acc / l;
    }
}

template <int BLOCK>
__global__ void kv_cache_attention_decode_kernel(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    int kv_len, int num_heads, int num_kv_heads, int head_dim) {
    int qh = blockIdx.x;
    int tid = threadIdx.x;

    int group = num_heads / num_kv_heads;
    int kvh = qh / group;
    size_t kv_hidden = static_cast<size_t>(num_kv_heads) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + static_cast<size_t>(qh) * head_dim;
    float* out_vec = out + static_cast<size_t>(qh) * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int pos = 0; pos < kv_len; ++pos) {
        size_t base = static_cast<size_t>(pos) * kv_hidden +
                      static_cast<size_t>(kvh) * head_dim;
        const float* k_vec = k_cache + base;
        const float* v_vec = v_cache + base;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_vec[d] * k_vec[d];
        }
        float score = block_reduce_sum<BLOCK>(partial) * scale;

        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        if (tid < head_dim) {
            acc = acc * alpha + beta * v_vec[tid];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (tid < head_dim) {
        out_vec[tid] = acc / l;
    }
}

} // namespace

template <typename W>
Status sgemm_impl(const float* A, const W* B, float* C, int M, int N, int K,
                  const char* ctx) {
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));

    size_t gx = (static_cast<size_t>(N) + 15) / 16;
    size_t gy = (static_cast<size_t>(M) + 15) / 16;
    RETURN_IF_ERROR(require_grid_x(gx, "sgemm grid.x"));
    RETURN_IF_ERROR(require_grid_x(gy, "sgemm grid.y"));

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));
    sgemm_kernel<W><<<grid, block>>>(A, B, C, M, N, K);
    return launch_status("cuda sgemm launch failed");
}

Status sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_impl(A, B, C, M, N, K, "sgemm");
}

Status sgemm(const float* A, const uint16_t* B_bf16, float* C, int M, int N, int K) {
    return sgemm_impl(A, B_bf16, C, M, N, K, "sgemm_bf16_weight");
}

template <typename W>
Status sgemm_nt_impl(const float* A, const W* B, float* C, int M, int N, int K,
                     const char* ctx) {
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));

    size_t gx = (static_cast<size_t>(N) + 15) / 16;
    size_t gy = (static_cast<size_t>(M) + 15) / 16;
    RETURN_IF_ERROR(require_grid_x(gx, "sgemm_nt grid.x"));
    RETURN_IF_ERROR(require_grid_x(gy, "sgemm_nt grid.y"));

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));
    sgemm_nt_kernel<W><<<grid, block>>>(A, B, C, M, N, K);
    return launch_status("cuda sgemm_nt launch failed");
}

Status sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_nt_impl(A, B, C, M, N, K, "sgemm_nt");
}

Status sgemm_nt(const float* A, const uint16_t* B_bf16, float* C, int M, int N, int K) {
    return sgemm_nt_impl(A, B_bf16, C, M, N, K, "sgemm_nt_bf16_weight");
}

Status add(const float* a, const float* b, float* y, int n) {
    if (n < 0) return Status::invalid_argument("cuda add requires non-negative n");
    if (n == 0) return Status::make_ok();
    RETURN_IF_ERROR(require_all_non_null("add", a, "a", b, "b", y, "y"));
    constexpr int block = 256;
    return launch_1d_grid(static_cast<size_t>(n), block, "add", [&](unsigned gx) {
        add_kernel<<<gx, block>>>(a, b, y, static_cast<size_t>(n));
        return launch_status("cuda add launch failed");
    });
}

template <typename W>
Status add_bias_impl(float* y, const W* bias, int rows, int cols, const char* ctx) {
    RETURN_IF_ERROR(require_all_non_null(ctx, y, "y", bias, "bias"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{rows, "rows"}, {cols, "cols"}}));

    size_t n = 0;
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(rows), static_cast<size_t>(cols), n, "add_bias elements"));
    constexpr int block = 256;
    return launch_1d_grid(n, block, ctx, [&](unsigned gx) {
        add_bias_kernel<W><<<gx, block>>>(y, bias, cols, n);
        return launch_status("cuda add_bias launch failed");
    });
}

Status add_bias(float* y, const float* bias, int rows, int cols) {
    return add_bias_impl(y, bias, rows, cols, "add_bias");
}

Status add_bias(float* y, const uint16_t* bias_bf16, int rows, int cols) {
    return add_bias_impl(y, bias_bf16, rows, cols, "add_bias_bf16");
}

Status mul(const float* a, const float* b, float* y, int n) {
    if (n < 0) return Status::invalid_argument("cuda mul requires non-negative n");
    if (n == 0) return Status::make_ok();
    RETURN_IF_ERROR(require_all_non_null("mul", a, "a", b, "b", y, "y"));
    constexpr int block = 256;
    return launch_1d_grid(static_cast<size_t>(n), block, "mul", [&](unsigned gx) {
        mul_kernel<<<gx, block>>>(a, b, y, static_cast<size_t>(n));
        return launch_status("cuda mul launch failed");
    });
}

Status silu(const float* x, float* y, int n) {
    if (n < 0) return Status::invalid_argument("cuda silu requires non-negative n");
    if (n == 0) return Status::make_ok();
    RETURN_IF_ERROR(require_all_non_null("silu", x, "x", y, "y"));
    constexpr int block = 256;
    return launch_1d_grid(static_cast<size_t>(n), block, "silu", [&](unsigned gx) {
        silu_kernel<<<gx, block>>>(x, y, static_cast<size_t>(n));
        return launch_status("cuda silu launch failed");
    });
}

Status fused_silu_mul(const float* gate, const float* up, float* y, int n) {
    if (n < 0) return Status::invalid_argument("cuda swiglu requires non-negative n");
    if (n == 0) return Status::make_ok();
    RETURN_IF_ERROR(require_all_non_null("swiglu", gate, "gate", up, "up", y, "y"));
    constexpr int block = 256;
    return launch_1d_grid(static_cast<size_t>(n), block, "swiglu", [&](unsigned gx) {
        swiglu_kernel<<<gx, block>>>(gate, up, y, static_cast<size_t>(n));
        return launch_status("cuda swiglu launch failed");
    });
}

template <typename W>
Status rmsnorm_impl(const float* x, const W* gamma, float* y,
                    int rows, int hidden, float eps, const char* ctx) {
    RETURN_IF_ERROR(require_all_non_null(ctx, x, "x", gamma, "gamma", y, "y"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{rows, "rows"}, {hidden, "hidden"}}));

    size_t elems = 0;
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(rows), static_cast<size_t>(hidden), elems, "rmsnorm elements"));
    constexpr int block = 256;
    rmsnorm_kernel<W, block><<<rows, block>>>(x, gamma, y, rows, hidden, eps);
    return launch_status("cuda rmsnorm launch failed");
}

Status rmsnorm(const float* x, const float* gamma, float* y,
               int rows, int hidden, float eps) {
    return rmsnorm_impl(x, gamma, y, rows, hidden, eps, "rmsnorm");
}

Status rmsnorm(const float* x, const uint16_t* gamma_bf16, float* y,
               int rows, int hidden, float eps) {
    return rmsnorm_impl(x, gamma_bf16, y, rows, hidden, eps, "rmsnorm_bf16_gamma");
}

template <typename W>
Status embedding_impl(const W* weight, const int* ids, float* out,
                      int seq_len, int vocab_size, int hidden, const char* ctx) {
    RETURN_IF_ERROR(require_all_non_null(ctx, weight, "weight", ids, "ids", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims(ctx,
                                {{seq_len, "seq_len"}, {vocab_size, "vocab_size"}, {hidden, "hidden"}}));

    size_t elems = 0;
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(vocab_size), static_cast<size_t>(hidden), elems, "embedding weight"));
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(seq_len), static_cast<size_t>(hidden), elems, "embedding output"));

    constexpr int block = 256;
    embedding_kernel<W, block><<<seq_len, block>>>(weight, ids, out, seq_len, vocab_size, hidden);
    return launch_status("cuda embedding launch failed");
}

Status embedding(const float* weight, const int* ids, float* out,
                 int seq_len, int vocab_size, int hidden) {
    return embedding_impl(weight, ids, out, seq_len, vocab_size, hidden, "embedding");
}

Status embedding(const uint16_t* weight_bf16, const int* ids, float* out,
                 int seq_len, int vocab_size, int hidden) {
    return embedding_impl(weight_bf16, ids, out, seq_len, vocab_size, hidden,
                          "embedding_bf16_weight");
}

Status apply_rope(const float* x, float* y, int tokens, int num_heads,
                  int head_dim, float base, int pos_offset) {
    RETURN_IF_ERROR(require_all_non_null("rope", x, "x", y, "y"));
    RETURN_IF_ERROR(validate_positive_dims("rope",
                                {{tokens, "tokens"}, {num_heads, "num_heads"}, {head_dim, "head_dim"}}));
    if (head_dim % 2 != 0)
        return Status::invalid_argument("cuda rope head_dim must be even");
    if (base <= 0.0f) return Status::invalid_argument("cuda rope base must be positive");
    if (pos_offset < 0) return Status::invalid_argument("cuda rope pos_offset must be non-negative");
    if (tokens - 1 > std::numeric_limits<int>::max() - pos_offset)
        return Status::unsupported("cuda rope position exceeds int range");

    size_t total = 0;
    RETURN_IF_ERROR(checked_mul3_size(static_cast<size_t>(tokens), static_cast<size_t>(num_heads),
                                      static_cast<size_t>(head_dim / 2), total, "rope elements"));
    constexpr int block = 256;
    return launch_1d_grid(total, block, "rope", [&](unsigned gx) {
        rope_kernel<<<gx, block>>>(x, y, tokens, num_heads, head_dim, base, pos_offset, total);
        return launch_status("cuda rope launch failed");
    });
}

Status softmax(const float* x, float* y, int rows, int cols) {
    RETURN_IF_ERROR(require_all_non_null("softmax", x, "x", y, "y"));
    RETURN_IF_ERROR(validate_positive_dims("softmax", {{rows, "rows"}, {cols, "cols"}}));

    constexpr int block = 256;
    softmax_kernel<block><<<rows, block>>>(x, y, rows, cols);
    return launch_status("cuda softmax launch failed");
}

Status transpose(const float* x, float* y, const int64_t* dims,
                 int rank, int axis0, int axis1) {
    RETURN_IF_ERROR(require_all_non_null("transpose", x, "x", y, "y", dims, "dims"));
    if (rank <= 0) return Status::invalid_argument("cuda transpose rank must be positive");
    if (axis0 < 0 || axis0 >= rank || axis1 < 0 || axis1 >= rank)
        return Status::invalid_argument("cuda transpose axes are out of range");

    size_t total = 1;
    for (int i = 0; i < rank; ++i) {
        if (dims[i] <= 0) return Status::invalid_argument("cuda transpose dims must be positive");
        RETURN_IF_ERROR(checked_mul_size(total, static_cast<size_t>(dims[i]), total, "transpose elements"));
    }

    int64_t* d_dims = nullptr;
    cudaError_t err = cudaMalloc(&d_dims, static_cast<size_t>(rank) * sizeof(int64_t));
    if (err != cudaSuccess) return cuda_status(err, "cuda transpose dims allocation failed");
    err = cudaMemcpy(d_dims, dims, static_cast<size_t>(rank) * sizeof(int64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_dims);
        return cuda_status(err, "cuda transpose dims copy failed");
    }

    constexpr int block = 256;
    size_t grid_x = (total + block - 1) / block;
    Status st = require_grid_x(grid_x, "transpose grid.x");
    if (!st.ok()) {
        cudaFree(d_dims);
        return st;
    }
    transpose_kernel<<<static_cast<unsigned>(grid_x), block>>>(
        x, y, d_dims, rank, axis0, axis1, total);
    cudaError_t sync_err = cudaDeviceSynchronize();
    cudaError_t free_err = cudaFree(d_dims);
    RETURN_IF_ERROR(cuda_status(sync_err, "cuda transpose kernel failed"));
    return cuda_status(free_err, "cuda transpose dims free failed");
}

Status sdpa(const float* q, const float* k, const float* v, float* out,
            int batch, int q_len, int num_heads, int num_kv_heads,
            int head_dim, bool causal) {
    RETURN_IF_ERROR(require_all_non_null("sdpa", q, "q", k, "k", v, "v", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims("sdpa", {{batch, "batch"}, {q_len, "q_len"},
                                                    {num_heads, "num_heads"}, {num_kv_heads, "num_kv_heads"},
                                                    {head_dim, "head_dim"}}));
    if (num_heads % num_kv_heads != 0)
        return Status::invalid_argument("cuda sdpa num_heads must be divisible by num_kv_heads");
    if (head_dim > 256)
        return Status::unsupported("cuda sdpa currently supports head_dim <= 256");

    size_t elems = 0;
    RETURN_IF_ERROR(checked_mul3_size(static_cast<size_t>(batch), static_cast<size_t>(q_len),
                                      static_cast<size_t>(num_heads) * head_dim, elems, "sdpa q/out elements"));
    RETURN_IF_ERROR(checked_mul3_size(static_cast<size_t>(batch), static_cast<size_t>(q_len),
                                      static_cast<size_t>(num_kv_heads) * head_dim, elems, "sdpa k/v elements"));

    constexpr int block = 256;
    dim3 grid(q_len, batch, num_heads);
    sdpa_kernel<block><<<grid, block>>>(q, k, v, out, batch, q_len,
                                        num_heads, num_kv_heads, head_dim, causal);
    return launch_status("cuda sdpa launch failed");
}

Status paged_attention_decode(const float* q, const float* k_cache,
                              const float* v_cache, const int* block_table,
                              float* out, int seq_len, int num_heads,
                              int num_kv_heads, int head_dim, int block_size) {
    RETURN_IF_ERROR(require_all_non_null("paged attention", q, "q", k_cache, "k_cache",
                                         v_cache, "v_cache", block_table, "block_table", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims("paged attention", {{seq_len, "seq_len"}, {num_heads, "num_heads"},
                                                              {num_kv_heads, "num_kv_heads"},
                                                              {head_dim, "head_dim"}, {block_size, "block_size"}}));
    if (num_heads % num_kv_heads != 0)
        return Status::invalid_argument("cuda paged attention num_heads must be divisible by num_kv_heads");
    if (head_dim > 256)
        return Status::unsupported("cuda paged attention currently supports head_dim <= 256");

    constexpr int block = 256;
    paged_attention_decode_kernel<block><<<num_heads, block>>>(
        q, k_cache, v_cache, block_table, out, seq_len, num_heads,
        num_kv_heads, head_dim, block_size);
    return launch_status("cuda paged attention decode launch failed");
}

Status paged_attention_decode_batch(const float* q, const float* k_cache,
                                    const float* v_cache, const int* block_tables,
                                    const int* sequence_lengths, float* out,
                                    int batch_size, int max_blocks_per_sequence,
                                    int num_heads, int num_kv_heads,
                                    int head_dim, int block_size) {
    RETURN_IF_ERROR(require_all_non_null("paged attention batch", q, "q", k_cache, "k_cache",
                                         v_cache, "v_cache", block_tables, "block_tables",
                                         sequence_lengths, "sequence_lengths", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims("paged attention batch",
                                {{batch_size, "batch_size"}, {max_blocks_per_sequence, "max_blocks_per_sequence"},
                                 {num_heads, "num_heads"}, {num_kv_heads, "num_kv_heads"},
                                 {head_dim, "head_dim"}, {block_size, "block_size"}}));
    if (num_heads % num_kv_heads != 0)
        return Status::invalid_argument("cuda paged attention batch num_heads must be divisible by num_kv_heads");
    if (head_dim > 256)
        return Status::unsupported("cuda paged attention batch currently supports head_dim <= 256");

    constexpr int block = 256;
    dim3 grid(num_heads, batch_size);
    paged_attention_decode_batch_kernel<block><<<grid, block>>>(
        q, k_cache, v_cache, block_tables, sequence_lengths, out,
        batch_size, max_blocks_per_sequence, num_heads, num_kv_heads,
        head_dim, block_size);
    return launch_status("cuda paged attention batch decode launch failed");
}

Status kv_cache_attention_decode(const float* q, const float* k_cache,
                                 const float* v_cache, float* out,
                                 int kv_len, int num_heads, int num_kv_heads,
                                 int head_dim) {
    RETURN_IF_ERROR(require_all_non_null("KV cache attention", q, "q", k_cache, "k_cache",
                                         v_cache, "v_cache", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims("KV cache attention", {{kv_len, "kv_len"}, {num_heads, "num_heads"},
                                                                  {num_kv_heads, "num_kv_heads"},
                                                                  {head_dim, "head_dim"}}));
    if (num_heads % num_kv_heads != 0)
        return Status::invalid_argument("cuda KV cache attention num_heads must be divisible by num_kv_heads");
    if (head_dim > 256)
        return Status::unsupported("cuda KV cache attention currently supports head_dim <= 256");

    constexpr int block = 256;
    kv_cache_attention_decode_kernel<block><<<num_heads, block>>>(
        q, k_cache, v_cache, out, kv_len, num_heads, num_kv_heads, head_dim);
    return launch_status("cuda KV cache attention decode launch failed");
}

} // namespace minillm::cuda
