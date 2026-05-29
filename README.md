# MiniLLMEngine

MiniLLMEngine is a compact C++23 CPU-first LLM inference project built for clarity. It implements the core pieces of a local inference runtime: tensor metadata, graph IR, shape inference, CPU kernel dispatch, optional CUDA kernel dispatch, GGUF parsing, weight loading, sampling, and KV cache based generation.

The goal is not to clone llama.cpp. The goal is to show the engineering ideas behind an inference engine in a smaller codebase that is easy to read, test, explain, and extend, with a shape that is friendly for internships and portfolio review.

## Highlights

- **C++23 inference runtime** with strongly typed graph IDs, `std::expected` error handling, and a small public umbrella header.
- **Graph IR + executor** with `Value`, `Node`, `GraphBuilder`, topological sort, backend capability checks, and runtime tensor binding.
- **CPU backend** for FP32 kernels including `Linear`, `MatMul`, `RMSNorm`, `QKNorm`, `RoPE`, `Attention`, `Softmax`, `SwiGLU`, `Embedding`, `Transpose`, and `Reshape`.
- **FlashAttention-style CPU attention** with tiled K/V traversal, online softmax, and a benchmark against the naive SDPA path.
- **Optional CUDA backend** behind `MINILLM_ENABLE_CUDA`, with FP32 kernels for the same core transformer operator set, Q8_0 weight-only kernels for model matrices, and a separate `CudaExecutor` path.
- **KV cache flow** for single-batch prefill/decode, including executor-driven cache length advancement.
- **Paged KV cache** with vLLM-style block tables, free-block reuse, CPU decode attention, multi-sequence scheduler, and CUDA device memory + paged decode over device block tables.
- **Continuous batching scheduler** with waiting→prefilling→decoding→finished lifecycle, auto-assigned sequence IDs, and KV block eviction/reuse.
- **Graph memory planner** with liveness analysis, O(n log n) best-fit buffer reuse, contiguous CPU/CUDA arena pools, and peak-memory reporting.
- **GGUF support** for bounds-checked metadata parsing, mmap-backed tensor byte views, F32/F16/BF16/Q8_0 weight loading, native BF16 weights, CPU/CUDA Q8_0 raw-block weight-only kernels, shared prefill/decode weight storage, tied-embedding aliases, and common Llama/Qwen weight-name mapping.
- **BPE tokenizer** with GPT-2 byte-to-unicode mapping, full regex pre-tokenization state machine, added-token longest-match, and `<0xHH>` byte token decoding.
- **Testing and benchmarks** with CTest, kernel reference tests, executor integration tests, and CPU GEMM/FlashAttention benchmarks.
- **Code quality** with `TRY`/`TRY_TENSOR` macros eliminating ~50 boilerplate error-propagation if-statements, `concepts` and `constexpr` replacing runtime helpers, `std::unreachable()` in exhaustive switches, and `kernel_adapter_common.h` deduplicating ~95 lines of shared adapter helpers.

## Architecture

```mermaid
flowchart LR
    GGUF["GGUF file"] --> Parser["GGUFParser"]
    Parser --> Loader["WeightLoader"]
    Loader --> Context["RuntimeContext"]

    Builder["GraphBuilder"] --> Graph["Graph IR"]
    Graph --> Executor["CpuExecutor"]
    Graph -. "optional" .-> CudaExecutor["CudaExecutor"]
    Registry["KernelRegistry"] --> Executor
    Registry -. "CUDA kernels" .-> CudaExecutor
    Backend["CpuBackend"] --> Executor
    CudaBackend["CudaBackend"] -.-> CudaExecutor
    Context --> Executor
    Context -.-> CudaExecutor

    Executor --> Kernels["CPU kernels"]
    CudaExecutor -.-> CudaKernels["CUDA kernels"]
    Context --> KV["KVCache"]
    Context --> PagedKV["PagedKVCache"]
    PagedKV --> PagedScheduler["PagedAttentionScheduler"]
    Kernels --> Tensor["Tensor storage"]
    PagedKV --> PagedAttn["CPU paged decode attention"]
    PagedScheduler --> PagedBatch["padded block-table batch"]
    PagedBatch -.-> CudaPaged["CUDA batched paged decode"]
    CudaKernels -.-> Tensor
    Executor --> Sampler["Sampler"]
    ContinuousScheduler["ContinuousBatchScheduler"] -.-> PagedKV
```

## What Is Implemented

| Area | Status |
|------|--------|
| Tensor / Shape / DType / Device | Implemented |
| Graph IR / GraphBuilder / shape inference | Implemented |
| CPU executor and kernel registry | Implemented |
| FP32 CPU kernels | Implemented for core transformer ops |
| CPU FlashAttention-style SDPA | Implemented for prefill and decode reference paths |
| Optional CUDA executor/backend | Experimental, disabled by default |
| FP32 CUDA kernels | Implemented with CUDA correctness tests |
| CUDA Q8_0 weight-only kernels | Implemented for `Embedding`, `Linear`, and `MatMul` with FP32 accumulation |
| Graph memory planner | Implemented for CPU and CUDA intermediates, with O(n log n) matching and contiguous arena binding |
| GGUF metadata and tensor loading | Implemented for F32/F16/BF16/Q8_0, with parser safety checks, mmap tensor views, shared weight storage, native BF16, and CPU/CUDA Q8_0 raw-block weight-only loading |
| Byte-level BPE tokenizer | Implemented with GPT-2 pre-tokenization, merge-based BPE, and `<0xHH>` byte token support |
| KV cache prefill/decode | Implemented for single-batch generation |
| Paged KV cache / PagedAttention | CPU: implemented. CUDA: device memory, block table upload, single-sequence & batched paged decode, contiguous cache prefill/decode — all wired through adapter. |
| Continuous batching scheduler | Lifecycle core implemented; real inference-loop integration in progress |
| CPU benchmark harness | Implemented |

## Quick Start

Requirements:

- CMake 3.22+
- C++23 compiler, such as GCC 13+, Clang 17+, or MSVC 19.35+

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

Optional CUDA build:

```bash
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DMINILLM_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

CUDA is off by default. The normal build remains CPU-only and does not require a CUDA toolkit or GPU. The CUDA build defaults to `sm_86` for Ampere GPUs such as RTX 30-series; pass `-DCMAKE_CUDA_ARCHITECTURES=...` to target another GPU.

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run examples:

```bash
./build/generate /path/to/model.gguf "Hello"
./build/generate_paged /path/to/model.gguf 32
./build/benchmark_cpu 1 512 512 5
./build/benchmark_flash_attention 2 8
./build-cuda/benchmark_cuda
./build-cuda/generate_cuda /path/to/model.gguf "Hello" 2
```

## Precision Modes

MiniLLMEngine keeps activations, logits, and KV cache in FP32 on the CPU path. Weights can use lower-precision physical storage when the kernel knows how to read them:

| GGUF dtype | Runtime storage | Compute path | Notes |
|------------|-------------|------------------|-------|
| F32 | `Float32` | FP32 load + FP32 accumulate | Baseline path |
| F16 | `Float32` | Expanded to FP32 at load | Kept simple and safe |
| BF16 | `BFloat16` | Weight-only BF16 read + FP32 activation/output | Default Qwen BF16 path |
| Q8_0 | `Q8_0` raw blocks on CPU/CUDA | Weight-only dequant inside `Embedding`/`Linear`/`MatMul`, FP32 accumulate | Packed dimensions used by the kernel must be multiples of 32 |

The Q8_0 path is true weight-only quantization: the shared weight tensor stores the original GGML `block_q8_0` bytes instead of expanding them at load time. CPU kernels keep the reference path small and readable. CUDA kernels read raw Q8 blocks on device, dequantize with the per-block FP16 scale, multiply by FP32 activations, and reduce into FP32 outputs. More aggressive SIMD and CUDA micro-kernel tuning are future extensions.

CUDA currently keeps BF16 GGUF weights native and Q8_0 GGUF weights raw on device; other formats stage through FP32 unless a dedicated CUDA kernel path exists.

## llama.cpp Alignment Check

To compare MiniLLMEngine against `llama.cpp` on the same Qwen3 prompt, use the
alignment helper:

```bash
python3 scripts/compare_llama_completion.py /path/to/Qwen3-0.6B-BF16.gguf "Hello" 32
```

It uses the raw chat prompt with the empty `<think></think>` block and the same
deterministic greedy decoding settings as the comparison path in `generate`.
That keeps the prompt shape and sampling behavior close to `llama-completion`
for end-to-end verification.

## CUDA Status

CUDA support is an optional experimental module, inspired by the operator structure in `mini_op` but integrated through MiniLLMEngine's own graph runtime:

- `CudaBackend` declares backend capability.
- `CudaExecutor` mirrors the CPU executor and dispatches `(DeviceType::CUDA, OpType)` kernels.
- `register_cuda_kernels()` bridges graph nodes to `.cu` launch wrappers.
- `Tensor::allocate_cuda()` owns element-typed CUDA device memory, while `Tensor::allocate_cuda_bytes()` owns raw device bytes for formats such as GGUF Q8_0 blocks.
- `WeightLoader` can stage F32/F16/BF16 GGUF weights through CPU memory, keep BF16 native, and copy Q8_0 raw block bytes directly into CUDA tensors.
- `SharedWeightStore` loads each GGUF tensor once and binds the same weight tensors into prefill and decode contexts.
- `KVCache::init_cuda()` stores contiguous decode K/V cache on device and lets CUDA Attention run prefill/decode.

Implemented CUDA operators currently target FP32 activations and outputs: `Embedding`, `Linear`, `MatMul`, `RMSNorm`, `QKNorm`, `Add`, `Mul`, `SiLU`, `SwiGLU`, `RoPE`, no-cache `Attention`, single-sequence and batched PagedAttention-style decode, `Softmax`, `Reshape`, and `Transpose`. FP32 `Linear`/`MatMul` dispatch through cuBLAS row-major adapters, while the original handwritten FP32 kernels remain available as `sgemm_reference()` and `sgemm_nt_reference()` for learning and debugging. No-cache CUDA attention uses a shared-memory tiled, online-softmax FlashAttention-style kernel. `Embedding`, `Linear`, and `MatMul` also accept `Q8_0` weight tensors and dequantize raw GGUF blocks inside warp/tile kernels.

`test_cuda_kernels` validates raw CUDA kernels against CPU references, including cuBLAS-backed `sgemm`/`sgemm_nt`, retained handwritten `sgemm_reference`/`sgemm_nt_reference`, tiled CUDA SDPA over multiple K/V tiles, Q8_0 `Embedding`, `sgemm_nt`, `sgemm`, executor `Linear`/`MatMul`, and defensive invalid-input paths.

`generate_cuda` is the current GPU smoke path for GGUF models. It builds the transformer graph on `Device::cuda(0)`, loads GGUF weights into CUDA tensors, runs prompt prefill, advances the shared cache once per graph run, then decodes one token at a time on GPU while sampling on CPU from copied logits.

The older forward-only demo binary was removed to keep the surface smaller. `test_cuda_kernels` covers the low-level CUDA operator surface, while `generate_cuda` covers the end-to-end GPU generation path.

The CUDA path does not yet include production scheduler policies, FA2-style multi-query tiling, or production-grade quantized GEMM tuning.

## Paged KV Cache

`PagedKVCache` is a compact implementation of the core idea behind PagedAttention:

- K/V memory is split into fixed-size token blocks (CPU via `std::vector`, CUDA via `cudaMalloc`).
- each sequence owns a block table that maps logical token positions to physical blocks.
- freed sequences return blocks to a reusable free list.
- `init_cuda()` allocates device-side block pools; `write_tokens_cuda()` copies K/V device-to-device; `upload_block_table()` stages the block table to device memory.
- `PagedAttentionScheduler` keeps a small active sequence set and emits padded batch metadata: sequence IDs, sequence lengths, and flattened block tables.
- `paged_attention_decode()` reads K/V through the block table and supports GQA (CPU reference).
- `cuda::paged_attention_decode()` runs the same decode pattern on device-resident K/V pages and a device block table.
- `cuda::paged_attention_decode_batch()` consumes scheduler-style batch metadata and decodes multiple active sequences in one launch.

The scheduler is intentionally small: it is not a production request queue, but it demonstrates the core bridge between paged KV allocation and batched GPU decode.

## Continuous Batching Scheduler

`ContinuousBatchScheduler` manages request lifecycle through four phases:

1. **Waiting**: request submitted but not yet allocated KV blocks
2. **Prefilling**: KV blocks allocated, prompt tokens being processed
3. **Decoding**: autoregressive token generation, one token per step
4. **Finished**: EOS reached or max tokens hit; blocks returned to free list

The caller drives the actual model execution. The scheduler provides:

- `submit()`: add a request with prompt tokens and max output length
- `admit_waiting()`: move waiting requests into active set (allocate KV blocks)
- `active_ids()`: which sequences are currently generating
- `evict_finished()`: free blocks for finished sequences
- `collect_finished()`: retrieve completed outputs
- `state()`: inspect per-sequence phase, tokens generated, etc.

This separation means the scheduler can be combined with any executor (CPU, CUDA, or mock) and any sampling strategy. The current code tests the scheduler lifecycle directly; wiring it through the real generation loop is the next implementation step.

```cpp
ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
scheduler.submit({.prompt_tokens = {1, 2, 3}, .max_tokens = 64});
scheduler.admit_waiting();
// ... run decode steps, call mark_token_generated(), mark_finished(), evict_finished() ...
```

## CPU Benchmarks

`benchmark_cpu` measures the GEMM paths used by the CPU backend.

```bash
./build/benchmark_cpu [M] [N] [K] [iters]
```

The `sgemm_nt` case matches the common transformer Linear layout:

```text
A[M,K] x W[N,K]^T -> C[M,N]
```

Release-mode run on the development server (24-core, AVX-512):

```text
./build-release/benchmark_cpu 1 2048 2048 20
sgemm_nt     shape=[1,2048,2048] iters=20 avg_ms=0.4983 gflops=16.84 (24 threads)
sgemm        shape=[1,2048,2048] iters=20 avg_ms=0.6069 gflops=13.82 (24 threads)

OMP_NUM_THREADS=1 ./build-release/benchmark_cpu 1 2048 2048 20
sgemm_nt     shape=[1,2048,2048] iters=20 avg_ms=0.2955 gflops=28.38 (1 thread)
sgemm        shape=[1,2048,2048] iters=20 avg_ms=0.5353 gflops=15.67 (1 thread)
```

| M (batch) | sgemm_nt (1 thr) | sgemm_nt (24 thr) | speedup |
|-----------|-------------------|---------------------|---------|
| 1 | 28.4 GFLOPS | 16.8 GFLOPS | 0.6x |
| 4 | — | 28.9 GFLOPS | — |
| 128 | — | 87.8 GFLOPS | — |

`sgemm_nt` is fastest single-threaded for M=1 decode on this machine because OpenMP overhead dominates the tiny batch. Larger prefill-shaped batches expose more parallel work and reach ~88 GFLOPS at M=128.

`benchmark_flash_attention` compares the naive CPU SDPA path against the tiled online-softmax FlashAttention-style CPU path:

```bash
./build/benchmark_flash_attention [iters] [heads]
```

```text
    seq head_dim  heads      sdpa_ms     flash_ms   speedup max_rel_err
    128       64      8        4.456        1.011      4.41    1.41e-04
    128      128      8       10.256        1.580      6.49    3.17e-03
    256       64      8       23.026        3.601      6.39    5.39e-04
    256      128      8       44.885        6.116      7.34    3.17e-03
    512       64      8       93.991       15.055      6.24    1.72e-03
    512      128      8      271.320       36.411      7.45    3.17e-03
```

FlashAttention-style CPU attention achieves about 4-7x speedup over naive SDPA across tested sequence lengths.

## CPU GGUF Smoke Tests

Release-mode CPU generation was smoke-tested with local Qwen3 GGUF files:

| Model | Prompt / max tokens | Result | Wall time | Peak RSS | Notes |
|-------|----------------------|--------|-----------|----------|-------|
| Qwen3-0.6B-BF16 | `Hello`, 8 | Passed | 2.02 s | 2.4 GB | shared BF16 weights: 1137 MiB |
| Qwen3-0.6B-BF16 | `Explain AI in one sentence`, 32 | Passed, stopped after 23 generated tokens | 3.96 s | 2.4 GB | EOS before max token limit |
| Qwen3-1.7B-BF16 | `Hello`, 4 | Passed | 7.20 s | 6.8 GB | shared BF16 weights: 3281 MiB |
| Qwen3-1.7B-BF16 | `Explain AI in one sentence`, 8 | Passed | 4.38 s | 6.8 GB | warm page cache after first run |
| Qwen3-0.6B-BF16 paged | 3 built-in prompts, 2 tokens each | Passed | 2.51 s | 3.3 GB | `generate_paged`, 6 KV blocks used |
| Qwen3-0.6B-Q8_0 | `Hello`, 8 | Passed | 29.53 s | 1.3 GB | llama.cpp Q8_0 GGUF, shared raw-block weights: 604 MiB |
| Qwen3-0.6B-Q8_0 | `Explain AI in one sentence`, 32 | Passed, stopped after 23 generated tokens | 83.84 s | 1.3 GB | EOS before max token limit |
| Qwen3-1.7B-Q8_0 | `Hello`, 4 | Passed | 66.71 s | 3.7 GB | llama.cpp Q8_0 GGUF, shared raw-block weights: 1743 MiB |
| Qwen3-0.6B-Q8_0 paged | 3 built-in prompts, 2 tokens each | Passed | 56.94 s | 2.2 GB | `generate_paged`, 6 KV blocks used |

The Q8_0 model files were produced with `llama.cpp/build/bin/llama-quantize ... Q8_0`. The 0.6B model quantized from 1137.00 MiB to 604.15 MiB in 1.93 s, and the 1.7B model quantized from 3281.97 MiB to 1743.77 MiB in 7.89 s. The CPU Q8_0 path currently prioritizes raw GGUF parity and memory reduction over speed; SIMD/tiled CPU Q8 dot kernels are still future optimization work.

## CPU/CUDA GGUF Alignment Smoke

The CPU and CUDA generation demos use the same Qwen chat prompt wrapper, including the empty `<think></think>` block used when thinking is disabled. Release-mode smoke tests on the local RTX 3080 16GB machine used the same prompt, max-token count, greedy sampler, and model file:

| Model | Backend | Prompt / max tokens | Output prefix | Wall time | Host peak RSS | Notes |
|-------|---------|----------------------|---------------|-----------|---------------|-------|
| Qwen3-0.6B-BF16 | CPU | `Hello`, 4 | `Hello! How can` | 1.60 s | 2.4 GB | shared BF16 weights: 1137 MiB |
| Qwen3-0.6B-BF16 | CUDA | `Hello`, 4 | `Hello! How can` | 3.80 s | 1.4 GB | shared CUDA BF16 weights: 1137 MiB |
| Qwen3-0.6B-Q8_0 | CPU | `Hello`, 4 | `Hello! How can` | 17.41 s | 1.3 GB | shared raw-block Q8_0 weights: 604 MiB |
| Qwen3-0.6B-Q8_0 | CUDA | `Hello`, 4 | `Hello! How can` | 1.50 s | 0.9 GB | shared CUDA raw-block Q8_0 weights: 604 MiB |

Low-level CUDA tests also compare cuBLAS FP32 GEMM, retained handwritten FP32 reference GEMM, tiled SDPA, Q8_0 `Embedding`, Q8_0 `Linear`/`MatMul`, executor dispatch, and defensive invalid-input paths against CPU references.

## CUDA Benchmarks

CUDA benchmarks measured on RTX 3080 16GB (Ampere SM86), FP32, Release build.

```bash
./build-cuda/benchmark_cuda
```

### Transformer ops

| Op | Shape | Time | Notes |
|----|-------|------|-------|
| `sgemm_nt` | [4, 4096] × [4096, 4096] | 0.14 ms | 943 GFLOPS via cuBLAS row-major adapter |
| `sgemm_nt` (decode) | [1, 4096] × [4096, 4096] | 0.14 ms | 243 GFLOPS via cuBLAS row-major adapter |
| `rmsnorm` | [4096, 1024] | 76 µs | — |
| `add` / `mul` | 16M elements | ~420 µs | Element-wise |
| `silu` | 16M elements | 281 µs | — |
| `fused_silu_mul` | 16M elements | 417 µs | SwiGLU fused |
| `embedding` | seq=256, vocab=151936 | 2.9 µs | Gather from 151936×1024 table |
| `apply_rope` | 256 tokens, 16 heads | 10.5 µs | In-place rotary embedding |
| `softmax` | [4096, 151936] | skipped by default cap | Per-row over full vocab |
| `sdpa` | seq=256, 16 heads, dim=128 | 1.64 ms | Shared-memory tiled causal SDPA with GQA |
| `kv_cache_attn` | KV=128 tokens, 16 heads | 88.7 µs | Contiguous cache decode |
| `paged_attn_decode` | KV=128 tokens, 16 heads | 103.6 µs | Paged cache decode (block_size=16) |
| `transpose` | [64, 128] | 8.8 µs | — |

## CPU vs CUDA comparison

| Op | CPU (24 threads) | CUDA (RTX 3080) | Ratio |
|----|-------------------|------------------|-------|
| GEMM M=4 (prefill) | 28.9 GFLOPS | 943 GFLOPS | 33× |
| GEMM M=1 (decode) | 16.8 GFLOPS | 243 GFLOPS | 14× |
| Flash/SDPA (seq=256) | 6.12 ms | 1.64 ms | 3.7× |

## RTX 3090 Porting Notes

RTX 3090 is also Ampere (`sm_86`), so the CUDA build can use the same architecture setting:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j

cmake -B build-cuda-sm86 -DCMAKE_BUILD_TYPE=Release \
  -DMINILLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-sm86 -j
```

Recommended first checks after moving the repo and GGUF files to a 3090 24GB server:

```bash
nvidia-smi
ctest --test-dir build-release --output-on-failure
ctest --test-dir build-cuda-sm86 --output-on-failure
./build-release/generate /path/to/Qwen3-0.6B-BF16.gguf "Hello" 4
./build-cuda-sm86/generate_cuda /path/to/Qwen3-0.6B-BF16.gguf "Hello" 4
./build-release/generate /path/to/Qwen3-0.6B-Q8_0.gguf "Hello" 4
./build-cuda-sm86/generate_cuda /path/to/Qwen3-0.6B-Q8_0.gguf "Hello" 4
```

Do not copy old `build-*` directories between machines; rebuild them on the target server. GGUF model files are intentionally kept outside git and should be copied separately with `rsync -avP` or equivalent.

## Tests

CTest currently runs:

```bash
./build/test_shape
./build/test_tensor
./build/test_graph
./build/test_graph_builder
./build/test_runtime
./build/test_cpu_kernels
./build/test_cpu_kernels_bf16
./build/test_memory_planner
./build/test_paged_kv_cache
./build/test_paged_attention_scheduler
./build/test_continuous_batch_scheduler
./build/test_gguf_parser
./build/test_transformer_graph_builder
./build/test_bpe_tokenizer
./build/test_e2e_verification
```

The test suite covers:

- shape and tensor allocation behavior
- graph construction and validation
- CPU executor integration
- CPU kernel numerical reference checks, including BF16 weights, Q8_0 weight-only GEMM/embedding, and FlashAttention-style SDPA comparisons
- graph liveness, memory reuse planning, and CPU/CUDA arena binding
- transformer graph weight naming and RoPE metadata propagation
- tokenizer boundary behavior and GGUF parser safety checks
- KV cache prefill/decode advancement
- paged KV block allocation, scheduler batch metadata, and paged decode attention
- continuous batching scheduler: admission, eviction, phase transitions, block reuse
- contiguous vs paged KV cache numerical alignment (cosine similarity, max abs diff)
- CUDA elementwise, GEMM, Q8_0 weight-only GEMM/embedding, norm, RoPE, softmax, transpose, SDPA, single/batched paged decode, and executor dispatch
- GGUF parser, mmap tensor views, F16/BF16 conversion, native BF16 load, CPU/CUDA Q8_0 raw weight loading, and shared-weight binding
- CUDA single-batch GGUF generation smoke path through `generate_cuda`

## Project Layout

```text
include/minillm/
  core/        Tensor, Shape, DType, Device, Status
  graph/       Graph IR, Node, Value, attributes, shape inference
  runtime/     Backend, executor, kernel registry, memory planner, sampler, scheduler
  runtime/kernels/  CPU/CUDA kernel declarations, SIMD helpers, adapter layer
  runtime/cache/    KV cache, paged KV cache, paged attention scheduler
  io/          GGUF parser, weight loader, tokenizer
  model/       Transformer graph builder
  utils/       bfloat16 type

src/
  core/        Core runtime data structures
  graph/       Graph implementation and builder logic
  runtime/     CPU backend, optional CUDA backend, and execution paths
  runtime/kernels/  CPU/CUDA kernel implementations
  runtime/cache/    KV cache implementations
  io/          GGUF and tokenizer implementation
  model/       Decoder-only graph construction

examples/      CLI demos, generation, and benchmark
tools/         Utility programs (dump_tokens)
tests/         Unit, integration, and E2E verification tests
docs/          Design notes
scripts/       Alignment helper scripts
```

## Design Notes

For a deeper explanation of the architecture, see [docs/design.md](docs/design.md). For runtime flow diagrams, see [docs/runtime_flows.md](docs/runtime_flows.md).

Key design choices:

- `Value` is a logical tensor descriptor. `Tensor` owns or references runtime storage.
- `GraphBuilder` performs shape inference when building nodes.
- `CpuExecutor` validates backend support, resolves kernels through `KernelRegistry`, and runs nodes in topological order.
- `CudaExecutor` mirrors the CPU executor when the project is built with `MINILLM_ENABLE_CUDA=ON`.
- `RuntimeContext` binds `ValueId` to runtime `Tensor` objects and owns intermediate tensors.
- `KVCache` is shared between prefill and decode contexts and is advanced once after a successful graph run.
- `SharedWeightStore` lets prefill and decode contexts reuse one loaded GGUF weight set instead of duplicating model parameters.
- `PagedKVCache` separates logical sequence positions from physical KV blocks; `PagedAttentionScheduler` turns several active sequences into padded block-table batches.
- `MemoryPlanner` computes intermediate tensor live ranges; `RuntimeContext::allocate_intermediates_planned()` binds non-overlapping CPU/CUDA intermediates to shared arena buffers.
- CUDA currently covers FP32 operator dispatch, tensor allocation, cuBLAS-backed FP32 GEMM, shared-memory tiled FlashAttention-style no-cache SDPA, native BF16 GGUF weight copies, raw Q8_0 GGUF weight copies with warp/tile `Embedding`/`Linear`/`MatMul` kernels, FP32 staging for other weight formats, contiguous CUDA KV cache generation, paged decode kernels with full adapter integration, and graph-memory arena allocation for intermediate tensors (both CPU and CUDA). Production batching policy, FA2-level attention tiling, and more specialized quantized kernels are intentionally left as future work.

## References

This project is an independent learning implementation inspired by:

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
- [Genllm](https://github.com/Aimol-l/Genllm)
- [mini_op](https://github.com/plutoaac/mini_op)

## Roadmap

Near-term work with high portfolio value:

- ~~Run and document end-to-end Qwen3-0.6B CPU and CUDA generation smoke demos with reference alignment.~~ Done.
- ~~Add Release-mode benchmark tables for prefill/decode latency and GEMM throughput.~~ Done.
- ~~Add Release-mode CUDA benchmark numbers for the tested FP32 kernels.~~ Done.
- ~~Implement the first quantized weight path: CPU `Q8_0` weight-only loading and kernels.~~ Done.
- ~~Implement CUDA `Q8_0` raw-block weight-only kernels for GGUF model matrices.~~ Done.
- ~~Move FP32 CUDA GEMM to cuBLAS and add a shared-memory tiled CUDA SDPA path.~~ Done.
- Connect `ContinuousBatchScheduler` to a real decode loop.

Longer-term experiments:

- More optimized GEMM micro-kernels, CPU/CUDA Q8_0 dot kernels, and weight packing.
- Multi-threaded CPU execution.
- Prefix cache, production continuous batching, and multi-sequence scheduling.
- Minimal streaming HTTP API.
