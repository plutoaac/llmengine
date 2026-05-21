#include "minillm/runtime/cuda_kernels.h"

#include <cfloat>
#include <cstddef>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

namespace minillm::cuda {

namespace {

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

size_t ceil_div_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

Status cuda_status(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return Status::make_ok();
    return Status::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

Status launch_status(const char* what) {
    return cuda_status(cudaGetLastError(), what);
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

__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void add_bias_kernel(float* y, const float* bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n) y[i] += bias[i % cols];
}

__global__ void mul_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

__global__ void silu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(x[i]);
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = silu_f(gate[i]) * up[i];
}

__global__ void sgemm_kernel(const float* A, const float* B, float* C,
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
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

__global__ void sgemm_nt_kernel(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = acc;
}

template <int BLOCK>
__global__ void rmsnorm_kernel(const float* x, const float* gamma, float* y,
                               int rows, int hidden, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * hidden;
    float* y_row = y + row * hidden;

    float local_sumsq = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        float v = x_row[h];
        local_sumsq += v * v;
    }
    float sumsq = block_reduce_sum<BLOCK>(local_sumsq);
    float inv_rms = rsqrtf(sumsq / hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        y_row[h] = x_row[h] * inv_rms * gamma[h];
    }
}

template <int BLOCK>
__global__ void embedding_kernel(const float* weight, const int* ids, float* out,
                                 int seq_len, int vocab_size, int hidden) {
    int s = blockIdx.x;
    if (s >= seq_len) return;
    int id = ids[s];
    float* dst = out + s * hidden;
    if (id < 0 || id >= vocab_size) {
        for (int h = threadIdx.x; h < hidden; h += BLOCK) {
            dst[h] = 0.0f;
        }
        return;
    }
    const float* src = weight + id * hidden;
    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        dst[h] = src[h];
    }
}

__global__ void rope_kernel(const float* x, float* y, int tokens, int num_heads,
                            int head_dim, float base, int pos_offset) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * num_heads * half;
    if (idx >= total) return;

    int d = idx % half;
    int h = (idx / half) % num_heads;
    int t = idx / (half * num_heads);
    int hidden = num_heads * head_dim;
    int base_idx = t * hidden + h * head_dim;
    int pos = pos_offset + t;

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
    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

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
    int q_hidden = num_heads * head_dim;
    int kv_hidden = num_kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + b * q_len * q_hidden + q_pos * q_hidden + qh * head_dim;
    float* out_vec = out + b * q_len * q_hidden + q_pos * q_hidden + qh * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int key_pos = 0; key_pos < q_len; ++key_pos) {
        if (causal && key_pos > q_pos) continue;
        const float* k_vec = k + b * q_len * kv_hidden + key_pos * kv_hidden + kvh * head_dim;
        const float* v_vec = v + b * q_len * kv_hidden + key_pos * kv_hidden + kvh * head_dim;

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
    int kv_hidden = num_kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    const float* q_vec = q + qh * head_dim;
    float* out_vec = out + qh * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        int physical_block = block_table[pos / block_size];
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

} // namespace

Status sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
    sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    return launch_status("cuda sgemm launch failed");
}

Status sgemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
    sgemm_nt_kernel<<<grid, block>>>(A, B, C, M, N, K);
    return launch_status("cuda sgemm_nt launch failed");
}

Status add(const float* a, const float* b, float* y, int n) {
    int block = 256;
    add_kernel<<<ceil_div(n, block), block>>>(a, b, y, n);
    return launch_status("cuda add launch failed");
}

Status add_bias(float* y, const float* bias, int rows, int cols) {
    int block = 256;
    add_bias_kernel<<<ceil_div(rows * cols, block), block>>>(y, bias, rows, cols);
    return launch_status("cuda add_bias launch failed");
}

Status mul(const float* a, const float* b, float* y, int n) {
    int block = 256;
    mul_kernel<<<ceil_div(n, block), block>>>(a, b, y, n);
    return launch_status("cuda mul launch failed");
}

Status silu(const float* x, float* y, int n) {
    int block = 256;
    silu_kernel<<<ceil_div(n, block), block>>>(x, y, n);
    return launch_status("cuda silu launch failed");
}

Status fused_silu_mul(const float* gate, const float* up, float* y, int n) {
    int block = 256;
    swiglu_kernel<<<ceil_div(n, block), block>>>(gate, up, y, n);
    return launch_status("cuda swiglu launch failed");
}

Status rmsnorm(const float* x, const float* gamma, float* y,
               int rows, int hidden, float eps) {
    constexpr int block = 256;
    rmsnorm_kernel<block><<<rows, block>>>(x, gamma, y, rows, hidden, eps);
    return launch_status("cuda rmsnorm launch failed");
}

Status embedding(const float* weight, const int* ids, float* out,
                 int seq_len, int vocab_size, int hidden) {
    constexpr int block = 256;
    embedding_kernel<block><<<seq_len, block>>>(weight, ids, out, seq_len,
                                                vocab_size, hidden);
    return launch_status("cuda embedding launch failed");
}

Status apply_rope(const float* x, float* y, int tokens, int num_heads,
                  int head_dim, float base, int pos_offset) {
    int total = tokens * num_heads * (head_dim / 2);
    int block = 256;
    rope_kernel<<<ceil_div(total, block), block>>>(x, y, tokens, num_heads,
                                                   head_dim, base, pos_offset);
    return launch_status("cuda rope launch failed");
}

Status softmax(const float* x, float* y, int rows, int cols) {
    constexpr int block = 256;
    softmax_kernel<block><<<rows, block>>>(x, y, rows, cols);
    return launch_status("cuda softmax launch failed");
}

Status transpose(const float* x, float* y, const int64_t* dims,
                 int rank, int axis0, int axis1) {
    if (rank <= 0) return Status::invalid_argument("cuda transpose rank must be positive");
    size_t total = 1;
    for (int i = 0; i < rank; ++i) total *= static_cast<size_t>(dims[i]);

    int64_t* d_dims = nullptr;
    cudaError_t err = cudaMalloc(&d_dims, static_cast<size_t>(rank) * sizeof(int64_t));
    if (err != cudaSuccess) return cuda_status(err, "cuda transpose dims allocation failed");
    err = cudaMemcpy(d_dims, dims, static_cast<size_t>(rank) * sizeof(int64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_dims);
        return cuda_status(err, "cuda transpose dims copy failed");
    }

    int block = 256;
    transpose_kernel<<<static_cast<unsigned>(ceil_div_size(total, static_cast<size_t>(block))), block>>>(
        x, y, d_dims, rank, axis0, axis1, total);
    Status st = launch_status("cuda transpose launch failed");
    cudaError_t free_err = cudaFree(d_dims);
    if (!st.ok()) return st;
    return cuda_status(free_err, "cuda transpose dims free failed");
}

Status sdpa(const float* q, const float* k, const float* v, float* out,
            int batch, int q_len, int num_heads, int num_kv_heads,
            int head_dim, bool causal) {
    if (head_dim > 256) {
        return Status::unsupported("cuda sdpa currently supports head_dim <= 256");
    }
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
    if (seq_len <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || block_size <= 0) {
        return Status::invalid_argument(
            "cuda paged attention requires positive seq_len, heads, head_dim, and block_size");
    }
    if (num_heads % num_kv_heads != 0) {
        return Status::invalid_argument(
            "cuda paged attention num_heads must be divisible by num_kv_heads");
    }
    if (head_dim > 256) {
        return Status::unsupported("cuda paged attention currently supports head_dim <= 256");
    }

    constexpr int block = 256;
    paged_attention_decode_kernel<block><<<num_heads, block>>>(
        q, k_cache, v_cache, block_table, out, seq_len, num_heads,
        num_kv_heads, head_dim, block_size);
    return launch_status("cuda paged attention decode launch failed");
}

} // namespace minillm::cuda
