#include "minillm/runtime/kernels/cuda_kernels.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cublas_v2.h>
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

Status cublas_status(cublasStatus_t err, const char* what) {
    if (err == CUBLAS_STATUS_SUCCESS) return Status::make_ok();
    return Status::runtime_error(
        std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(err)));
}

Status cublas_handle(cublasHandle_t& out) {
    struct Handle {
        cublasHandle_t h{nullptr};
        ~Handle() {
            if (h) (void)cublasDestroy(h);
        }
    };
    thread_local Handle handle;
    if (!handle.h) {
        auto st = cublasCreate(&handle.h);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return cublas_status(st, "cublasCreate failed");
        }
    }
    out = handle.h;
    return Status::make_ok();
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

__device__ __forceinline__ float f16_bits_to_f32(uint16_t bits) {
    const unsigned sign = (bits >> 15) & 0x1u;
    unsigned exp = (bits >> 10) & 0x1fu;
    unsigned mant = bits & 0x3ffu;
    unsigned f32 = 0;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            const int shift = __clz(mant) - 21;
            mant <<= shift;
            const int normalized_exp = 1 - shift;
            f32 = (sign << 31) |
                  (static_cast<unsigned>(normalized_exp + 112) << 23) |
                  ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7f800000u | (mant ? 0x400000u : 0u);
    } else {
        f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return __uint_as_float(f32);
}

__device__ __forceinline__ uint16_t load_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
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

template <int WARPS>
__global__ void sgemm_q8_0_nblock_kernel(const float* A, const uint8_t* B, float* C,
                                         int M, int N, int K) {
    constexpr int warp_size = 32;
    constexpr int block_bytes = 34;
    __shared__ float partials[WARPS][warp_size];

    const int lane = threadIdx.x & (warp_size - 1);
    const int warp = threadIdx.x >> 5;
    const int n_block = blockIdx.x;
    const int m = blockIdx.y;
    const int n = n_block * warp_size + lane;
    if (m >= M || n >= N) return;

    const float* a_row = A + static_cast<size_t>(m) * K;
    const int blocks_per_b_row = N / warp_size;
    float local = 0.0f;
    for (int k = warp; k < K; k += WARPS) {
        const uint8_t* block =
            B + (static_cast<size_t>(k) * blocks_per_b_row + n_block) * block_bytes;
        uint16_t scale_bits = 0;
        if (lane == 0) scale_bits = load_u16_le(block);
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0);
        const auto* qs = reinterpret_cast<const int8_t*>(block + 2);
        local += a_row[k] * static_cast<float>(qs[lane]) * f16_bits_to_f32(scale_bits);
    }
    partials[warp][lane] = local;
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        sum += partials[w][lane];
    }
    C[static_cast<size_t>(m) * N + n] = sum;
}

template <int WARPS>
__global__ void sgemm_nt_q8_0_warp_kernel(const float* A, const uint8_t* B, float* C,
                                          int M, int N, int K) {
    constexpr int warp_size = 32;
    constexpr int block_bytes = 34;
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * WARPS + warp;
    const int m = blockIdx.y;
    if (m >= M || n >= N) return;

    const float* a_row = A + static_cast<size_t>(m) * K;
    const uint8_t* b_row = B + static_cast<size_t>(n) * (K / warp_size) * block_bytes;
    float local = 0.0f;
    for (int qb = 0; qb < K / warp_size; ++qb) {
        const uint8_t* block = b_row + static_cast<size_t>(qb) * block_bytes;
        uint16_t scale_bits = 0;
        if (lane == 0) scale_bits = load_u16_le(block);
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0);
        const auto* qs = reinterpret_cast<const int8_t*>(block + 2);
        local += a_row[qb * warp_size + lane] *
                 static_cast<float>(qs[lane]) * f16_bits_to_f32(scale_bits);
    }
    const float sum = warp_reduce_sum(local);
    if (lane == 0) C[static_cast<size_t>(m) * N + n] = sum;
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

template <int WARPS>
__global__ void embedding_q8_0_warp_kernel(const uint8_t* weight, const int* ids, float* out,
                                           int seq_len, int vocab_size, int hidden) {
    constexpr int warp_size = 32;
    constexpr int block_bytes = 34;
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp = threadIdx.x >> 5;
    int s = blockIdx.x;
    if (s >= seq_len) return;
    int id = ids[s];
    float* dst = out + static_cast<size_t>(s) * hidden;
    if (id < 0 || id >= vocab_size) {
        for (int h = threadIdx.x; h < hidden; h += WARPS * warp_size) {
            dst[h] = 0.0f;
        }
        return;
    }
    const int blocks_per_row = hidden / warp_size;
    const uint8_t* row = weight + static_cast<size_t>(id) * blocks_per_row * block_bytes;
    for (int qb = warp; qb < blocks_per_row; qb += WARPS) {
        const uint8_t* block = row + static_cast<size_t>(qb) * block_bytes;
        uint16_t scale_bits = 0;
        if (lane == 0) scale_bits = load_u16_le(block);
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0);
        const auto* qs = reinterpret_cast<const int8_t*>(block + 2);
        dst[qb * warp_size + lane] =
            static_cast<float>(qs[lane]) * f16_bits_to_f32(scale_bits);
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

template <int BLOCK, int TILE>
__global__ void flash_sdpa_tiled_kernel(const float* q, const float* k, const float* v,
                                        float* out, int batch, int q_len,
                                        int num_heads, int num_kv_heads,
                                        int head_dim, bool causal) {
    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = q_s + head_dim;
    float* v_s = k_s + TILE * head_dim;
    float* scores = v_s + TILE * head_dim;

    const int q_pos = blockIdx.x;
    const int b = blockIdx.y;
    const int qh = blockIdx.z;
    const int tid = threadIdx.x;

    const int group = num_heads / num_kv_heads;
    const int kvh = qh / group;
    const size_t q_hidden = static_cast<size_t>(num_heads) * head_dim;
    const size_t kv_hidden = static_cast<size_t>(num_kv_heads) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + static_cast<size_t>(b) * q_len * q_hidden +
                         static_cast<size_t>(q_pos) * q_hidden +
                         static_cast<size_t>(qh) * head_dim;
    float* out_vec = out + static_cast<size_t>(b) * q_len * q_hidden +
                     static_cast<size_t>(q_pos) * q_hidden +
                     static_cast<size_t>(qh) * head_dim;

    for (int d = tid; d < head_dim; d += BLOCK) {
        q_s[d] = q_vec[d];
    }
    __syncthreads();

    const int valid_keys = causal ? (q_pos + 1) : q_len;
    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < valid_keys; tile_start += TILE) {
        const int tile_count = min(TILE, valid_keys - tile_start);
        const int tile_elems = tile_count * head_dim;

        for (int idx = tid; idx < tile_elems; idx += BLOCK) {
            const int t = idx / head_dim;
            const int d = idx - t * head_dim;
            const size_t base = static_cast<size_t>(b) * q_len * kv_hidden +
                                static_cast<size_t>(tile_start + t) * kv_hidden +
                                static_cast<size_t>(kvh) * head_dim + d;
            k_s[idx] = k[base];
            v_s[idx] = v[base];
        }
        __syncthreads();

        float tile_m = -FLT_MAX;
        for (int t = 0; t < tile_count; ++t) {
            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += BLOCK) {
                partial += q_s[d] * k_s[t * head_dim + d];
            }
            const float score = block_reduce_sum<BLOCK>(partial) * scale;
            if (tid == 0) scores[t] = score;
            tile_m = fmaxf(tile_m, score);
        }
        __syncthreads();

        const float new_m = fmaxf(m, tile_m);
        const float alpha = expf(m - new_m);
        float beta_sum = 0.0f;
        if (tid < head_dim) {
            acc *= alpha;
        }
        for (int t = 0; t < tile_count; ++t) {
            const float beta = expf(scores[t] - new_m);
            beta_sum += beta;
            if (tid < head_dim) {
                acc += beta * v_s[t * head_dim + tid];
            }
        }
        l = l * alpha + beta_sum;
        m = new_m;
        __syncthreads();
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

Status sgemm_reference(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_impl(A, B, C, M, N, K, "sgemm_reference");
}

Status sgemm_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    const char* ctx = "sgemm_cublas";
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));

    cublasHandle_t handle = nullptr;
    RETURN_IF_ERROR(cublas_handle(handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K, &alpha,
                          B, N,
                          A, K,
                          &beta, C, N);
    return cublas_status(st, "cublasSgemm row-major NN failed");
}

Status sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_cublas(A, B, C, M, N, K);
}

Status sgemm(const float* A, const uint16_t* B_bf16, float* C, int M, int N, int K) {
    return sgemm_impl(A, B_bf16, C, M, N, K, "sgemm_bf16_weight");
}

Status sgemm(const float* A, const uint8_t* B_q8_0, float* C, int M, int N, int K) {
    const char* ctx = "sgemm_q8_0_weight";
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B_q8_0, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    if (K % 32 != 0 || N % 32 != 0) {
        return Status::unsupported("cuda sgemm Q8_0 requires K and N to be multiples of 32");
    }
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));
    RETURN_IF_ERROR(require_grid_x(static_cast<size_t>(N), "sgemm_q8_0 grid.x"));
    RETURN_IF_ERROR(require_grid_x(static_cast<size_t>(M), "sgemm_q8_0 grid.y"));

    constexpr int warps = 8;
    constexpr int block = warps * 32;
    dim3 grid(static_cast<unsigned>(N / 32), static_cast<unsigned>(M));
    sgemm_q8_0_nblock_kernel<warps><<<grid, block>>>(A, B_q8_0, C, M, N, K);
    return launch_status("cuda sgemm Q8_0 launch failed");
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

Status sgemm_nt_reference(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_nt_impl(A, B, C, M, N, K, "sgemm_nt_reference");
}

Status sgemm_nt_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    const char* ctx = "sgemm_nt_cublas";
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));

    cublasHandle_t handle = nullptr;
    RETURN_IF_ERROR(cublas_handle(handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto st = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          N, M, K, &alpha,
                          B, K,
                          A, K,
                          &beta, C, N);
    return cublas_status(st, "cublasSgemm row-major NT failed");
}

Status sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    return sgemm_nt_cublas(A, B, C, M, N, K);
}

Status sgemm_nt(const float* A, const uint16_t* B_bf16, float* C, int M, int N, int K) {
    return sgemm_nt_impl(A, B_bf16, C, M, N, K, "sgemm_nt_bf16_weight");
}

Status sgemm_nt(const float* A, const uint8_t* B_q8_0, float* C, int M, int N, int K) {
    const char* ctx = "sgemm_nt_q8_0_weight";
    RETURN_IF_ERROR(require_all_non_null(ctx, A, "A", B_q8_0, "B", C, "C"));
    RETURN_IF_ERROR(validate_positive_dims(ctx, {{M, "M"}, {N, "N"}, {K, "K"}}));
    if (K % 32 != 0) {
        return Status::unsupported("cuda sgemm_nt Q8_0 requires K to be a multiple of 32");
    }
    RETURN_IF_ERROR(validate_gemm_sizes(M, N, K, ctx));
    RETURN_IF_ERROR(require_grid_x(static_cast<size_t>(N), "sgemm_nt_q8_0 grid.x"));
    RETURN_IF_ERROR(require_grid_x(static_cast<size_t>(M), "sgemm_nt_q8_0 grid.y"));

    constexpr int warps = 8;
    constexpr int block = warps * 32;
    const size_t gx = (static_cast<size_t>(N) + warps - 1) / warps;
    RETURN_IF_ERROR(require_grid_x(gx, "sgemm_nt_q8_0 grid.x"));
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(M));
    sgemm_nt_q8_0_warp_kernel<warps><<<grid, block>>>(A, B_q8_0, C, M, N, K);
    return launch_status("cuda sgemm_nt Q8_0 launch failed");
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

Status embedding(const uint8_t* weight_q8_0, const int* ids, float* out,
                 int seq_len, int vocab_size, int hidden) {
    const char* ctx = "embedding_q8_0_weight";
    RETURN_IF_ERROR(require_all_non_null(ctx, weight_q8_0, "weight", ids, "ids", out, "out"));
    RETURN_IF_ERROR(validate_positive_dims(ctx,
                                {{seq_len, "seq_len"}, {vocab_size, "vocab_size"}, {hidden, "hidden"}}));
    if (hidden % 32 != 0) {
        return Status::unsupported("cuda embedding Q8_0 requires hidden to be a multiple of 32");
    }

    size_t elems = 0;
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(vocab_size), static_cast<size_t>(hidden), elems, "embedding Q8_0 weight"));
    RETURN_IF_ERROR(checked_mul_size(static_cast<size_t>(seq_len), static_cast<size_t>(hidden), elems, "embedding Q8_0 output"));

    constexpr int warps = 8;
    constexpr int block = warps * 32;
    embedding_q8_0_warp_kernel<warps><<<seq_len, block>>>(
        weight_q8_0, ids, out, seq_len, vocab_size, hidden);
    return launch_status("cuda embedding Q8_0 launch failed");
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
    constexpr int tile = 16;
    dim3 grid(q_len, batch, num_heads);
    const size_t shared_floats =
        static_cast<size_t>(head_dim) + 2 * static_cast<size_t>(tile) * head_dim + tile;
    const size_t shared_bytes = shared_floats * sizeof(float);
    if (shared_bytes > 48 * 1024) {
        return Status::unsupported("cuda flash SDPA shared-memory tile exceeds 48 KiB");
    }
    flash_sdpa_tiled_kernel<block, tile><<<grid, block, shared_bytes>>>(
        q, k, v, out, batch, q_len, num_heads, num_kv_heads, head_dim, causal);
    return launch_status("cuda flash SDPA launch failed");
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
