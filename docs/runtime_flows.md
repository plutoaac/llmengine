# MiniLLMEngine Runtime Flows

Key execution paths in MiniLLMEngine, from model file to generated token.

## Model Loading (mmap Fast Path)

```mermaid
flowchart TD
    User["generate example"] --> WL["WeightLoader::open()"]
    WL --> Parse["GGUFParser::parse()"]
    Parse --> StreamRead["std::ifstream: parse header + metadata + tensor infos"]
    StreamRead --> Mmap["try_open_memory_view()"]
    Mmap -->|success| MV["PosixMappedGGUFMemoryView (mmap) or WindowsMappedGGUFMemoryView (MapViewOfFile)"]
    Mmap -->|fallback| NoMmap["memory_view = null"]
    MV --> GGUFFile["GGUFFile { memory_view, tensor_infos, data_offset }"]
    NoMmap --> GGUFFile

    GGUFFile --> Config["WeightLoader::extract_config()"]
    Config --> LoadWeights["WeightLoader::load_shared_weights()"]

    LoadWeights --> Loop["for each Constant value in graph:"]
    Loop --> TensorRaw["tensor_raw_view(tensor_info)"]
    TensorRaw -->|has memory_view| ViewSlice["file.tensor_view(ti) → span slice of mmap region"]
    TensorRaw -->|no memory_view| Fallback["read_tensor_data() → ifstream seek+read"]
    ViewSlice --> DTypeDispatch["WeightLoader dtype dispatch"]
    Fallback --> DTypeDispatch
    DTypeDispatch --> F32["F32 / F16 -> CPU Float32"]
    DTypeDispatch --> BF16["BF16 -> native BFloat16"]
    DTypeDispatch --> Q8["Q8_0 -> raw CPU block_q8_0"]
    F32 -. "CUDA non-BF16" .-> CudaF32["host F32 staging -> cudaMemcpy"]
    BF16 -. "CUDA BF16" .-> CudaBF16["raw BF16 cudaMemcpy"]
    F32 --> Store["SharedWeightStore"]
    BF16 --> Store
    Q8 --> Store
    CudaF32 --> Store
    CudaBF16 --> Store

    Store --> Bind["bind(graph, context): map value names → loaded Tensors"]
```

**Key design**: `GGUFMemoryView` is a cross-platform read-only memory view. On Linux it uses `mmap()` with `MAP_PRIVATE`. On Windows it uses `CreateFileMappingA` + `MapViewOfFile`. The `SharedWeightStore` holds canonical tensors; `RuntimeContext` holds non-owning pointers. CPU Q8_0 weights remain compressed in this store; CUDA Q8_0 still stages through FP32 until a CUDA quantized matmul path is added.

## Single-Sequence Generation (CPU)

```mermaid
flowchart TD
    Tokenizer["Tokenizer::encode(text) → token IDs"] --> Prefill["Prefill Graph (seq_len=N)"]
    Prefill --> Exec["CpuExecutor::run(ctx)"]

    Exec --> Embed["Embedding: ids → vectors"]
    Embed --> Layer["for each transformer layer:"]
    Layer --> QKV["Linear Q/K/V"]
    QKV --> RoPE["RoPE: apply rotary embedding"]
    RoPE --> Attn["Attention: write K/V to cache; attend over all history"]
    Attn --> Out["Output Linear"]
    Out --> FFN["RMSNorm → SiLU/Gate → Down"]
    FFN --> NextLayer["next layer"]
    NextLayer --> LMHead["RMSNorm → Linear(lm_head) → logits"]
    LMHead --> Sample["Sampler::sample(logits) → next token"]

    Sample --> Decode["Decode Graph (seq_len=1)"]
    Decode --> DecExec["CpuExecutor::run(ctx)"]
    DecExec --> DecAttn["Attention: append K/V, attend over cache"]
    DecAttn --> DecSample["Sample → next token"]
    DecSample -->|"not EOS"| Decode
    DecSample -->|"EOS or max_tokens"| Done["Return decoded text"]
```

## Multi-Sequence Paged Generation

```mermaid
flowchart TD
    Prompts["N prompts → N sequence IDs"] --> Setup["PagedKVCache: reserve blocks per sequence"]
    Setup --> PrefillLoop["for each sequence:"]
    PrefillLoop --> PrefillCtx["Contiguous KVCache prefill"]
    PrefillCtx --> Copy["Copy K/V to PagedKVCache via write_tokens()"]

    Copy --> DecodeLoop["Decode round-robin over N sequences"]
    DecodeLoop --> SetCtx["set_paged_sequence_id(seq)"]
    SetCtx --> DecExec2["CpuExecutor::run(ctx)"]
    DecExec2 --> RoPE2["RoPE: position from paged_kv_cache write_pos"]
    RoPE2 --> PageAttn["attention_paged: write K/V → paged cache; paged_attention_decode ()"]
    PageAttn --> Sample2["Sample"]
    Sample2 -->|"next seq"| DecodeLoop
    Sample2 -->|"all done"| Collect["Collect finished outputs"]

    Setup -.-> Scheduler["PagedAttentionScheduler: active set, padded block tables"]
```

## CUDA Generation Path

```mermaid
flowchart TD
    CudaSetup["cudaSetDevice(0); config.device = Device::cuda(0)"] --> CudaGraph["Build graph with CUDA device"]
    CudaGraph --> CudaWeights["load_shared_weights → dequantize → cudaMemcpy to GPU"]
    CudaWeights --> CudaCache["KVCache::init_cuda() → cudaMalloc K/V"]
    CudaCache --> CudaPrefill["CudaExecutor::run(prefill context)"]
    CudaPrefill --> CudaDecode["CudaExecutor::run(decode context)"]
    CudaDecode --> CudaSample["cudaMemcpy logits → CPU → Sample"]
    CudaSample -->|"loop"| CudaDecode

    CudaDecode -.-> CudaPaged["CUDA PagedAttention (via PagedKVCache::init_cuda + write_tokens_cuda)"]
```

## Quantized Weight Loading (Q8_0)

```mermaid
flowchart LR
    RawBytes["GGUF raw bytes: [scale:2B][int8 x 32]"] --> StoreQ8["Tensor dtype = Q8_0\nraw block_q8_0 storage"]
    Act["FP32 activation"] --> Kernel["CPU Embedding / Linear / MatMul"]
    StoreQ8 --> Kernel
    Kernel --> BlockLoop["per block: scale = f16_to_f32; q = int8"]
    BlockLoop --> Dot["acc += activation * (q * scale)"]
    Dot --> Out["FP32 output"]
```

Q8_0 stores weights as 8-bit integers with a per-block 16-bit float scale. On the CPU path MiniLLMEngine now keeps those packed bytes in `SharedWeightStore` and dequantizes inside the weight kernel while accumulating in FP32. That is true weight-only quantization: lower weight memory, unchanged FP32 activations and outputs.

This first path is intentionally minimal and readable. It requires the packed weight dimension to be a multiple of 32 and does not yet include SIMD dot kernels, Q4 formats, or CUDA quantized matmul.

## Tokenizer Flow

```mermaid
flowchart TD
    GGUF["tokenizer.ggml.tokens + merges from GGUF"] --> Init["BPETokenizer::init(GGUFFile)"]
    Init --> Vocab["vocab_[] + vocab_map_ + bpe_ranks_"]
    Vocab --> AddedTokens["pre_tokenize each token; split>1 → added_tokens_"]

    Text["input text"] --> Encode["encode(text)"]
    Encode --> AddedMatch["Longest-match added_tokens_ first"]
    AddedMatch --> PreTok["pre_tokenize: GPT-2 regex state machine → byte pieces"]
    PreTok --> ByteEnc["bytes_to_unicode: byte → GPT-2 unicode mapping"]
    ByteEnc --> BPE["bpe(): linked-list greedy merge by bpe_ranks_ priority"]
    BPE --> IDs["vocab_map_ lookup → token ID list"]

    IDs --> Decode["decode(ids)"]
    Decode --> TokText["token_to_text(id): handle <0xHH> byte tokens"]
    TokText --> ByteDec["unicode_to_bytes: unicode → raw byte"]
```
