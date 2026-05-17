# MiniLLMEngine

A C++23 lightweight LLM inference engine for learning and experimentation. The current phase focuses on **Tensor abstractions** and **computation graph IR**, laying the groundwork for future GGUF loading, CPU SIMD backends, KV Cache, and RPC serving.

## References

- **[Genllm](https://github.com/Aimol-l/Genllm)** — C++23 modular LLM inference framework: GGUF parsing, DAG computation graph IR, CPU/CUDA backends, memory pool, RoPE cache, paged attention, model factory.
- **[llama.cpp / ggml](https://github.com/ggerganov/llama.cpp)** — Tensor + op graph + context memory low-level abstraction.
- **[mini_op](https://github.com/plutoaac/mini_op)** — Future CPU SIMD kernel source (matmul, rmsnorm, rope, attention, elementwise, fused ops).

Note: This project is an independent learning implementation, not a fork of any of the above.

## Current Capabilities

- **Tensor / Shape / DType / Device** — Core tensor abstraction with dynamic dims, FP32/FP16/BF16/Int32/Int64/UInt8/Int8/Bool dtypes, CPU/CUDA/Metal device stubs.
- **Graph / Value / Node / Attribute** — DAG computation graph IR with strongly-typed IDs (ValueId, NodeId) and per-node attributes.
- **GraphBuilder** — Fluent API for constructing computation graphs with automatic shape inference.
- **Shape Inference** — MatMul, Linear, Add, Embedding, RMSNorm, RoPE, SiLU, SwiGLU, Attention shape propagation with dynamic dim support.
- **Transformer Graph Builder** — Builds a tiny decoder-only transformer block graph (Llama-style).
- **MockExecutor** — Validates, topologically sorts, and walks the graph without numerical computation.
- **CPU Backend Stub** — Declares supported ops; does not perform real computation yet.
- **KernelRegistry** — Pluggable kernel dispatch; currently supports mock kernels, designed for mini_op integration.

## Not In Current Scope

- GGUF file parsing
- Tokenizer
- Real numerical kernels (SIMD or otherwise)
- CUDA / Metal execution
- Quantization (Q4/Q8/GGML quantized dtypes)
- KV Cache
- Batch scheduling
- Paged attention

## Build

Requires a C++23 compiler (GCC 13+, Clang 17+, MSVC 19.35+).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

## Run Example

```bash
./build/examples/build_tiny_llama_graph
```

This builds a tiny Llama-style decoder block graph, validates it, dumps the full IR, then runs a mock execution pass.

## Run Tests

```bash
./build/tests/test_shape
./build/tests/test_tensor
./build/tests/test_graph
./build/tests/test_graph_builder
```

## Graph IR Design

| Concept | Role |
|---------|------|
| **Value** | Logical tensor descriptor (shape, dtype, device, kind). Does not hold data. |
| **Node** | Computation operator with inputs/outputs (ValueIds) and attributes. |
| **Graph** | DAG of Values and Nodes. Supports validation and topological sort. |
| **GraphBuilder** | Constructs Values + Nodes with automatic shape inference. |
| **Executor** | Compiles (validate + topo sort + backend check) and runs the graph. |
| **Backend** | Declares which OpTypes a device supports. |
| **KernelRegistry** | Maps (DeviceType, OpType) to kernel functions; future dispatch to mini_op. |

Key design decisions:
- **Value** is a logical tensor; **Tensor** is a physical data container. The Executor binds a Tensor to a ValueId at runtime.
- **ValueId / NodeId** are strongly typed — no raw `int` mixing.
- Dynamic dims (`-1`) propagate through shape inference; `numel()` and `allocate_cpu()` reject dynamic shapes.
- Attributes on Nodes carry op-specific metadata (RMSNorm eps, RoPE num_heads/head_dim, Attention causal flag).

## OpType to mini_op Mapping (Future)

When the CPU kernel adapter is implemented, the following mapping will connect Graph ops to mini_op SIMD kernels:

| OpType | mini_op Kernel |
|--------|---------------|
| `Linear` / `MatMul` | `cpu_gemm` |
| `RMSNorm` | `cpu_llm_ops::rms_norm` |
| `RoPE` | `cpu_rope` |
| `Attention` | `cpu_attention` |
| `Softmax` | `cpu_llm_ops::softmax` / `cpu_attention::softmax` |
| `Embedding` | `cpu_llm_ops::embedding` |
| `Add` / `Mul` / `SiLU` | `cpu_elementwise` |
| `SwiGLU` | `cpu_fused_ops::fused_silu_mul` |
| `Transpose` | `cpu_transpose` |

## Roadmap

| Phase | Goal |
|-------|------|
| 1 | Tensor + Graph IR + MockExecutor *(current)* |
| 2 | CPU kernel adapter — integrate mini_op SIMD ops |
| 3 | GGUF metadata / tensor loader |
| 4 | Llama/Qwen decoder-only model graph builder |
| 5 | Tokenizer + greedy generation |
| 6 | KV Cache + prefill/decode separation |
| 7 | Streaming generation + mini_RPC serving |
| 8 | Benchmark tokens/s, TTFT, memory usage |
| 9 | CUDA backend / quantization / paged attention |
