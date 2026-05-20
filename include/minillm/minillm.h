#pragma once

// Umbrella header for the MiniLLMEngine library.
// Include this for convenient access to all public APIs.

// Core
#include "minillm/core/status.h"
#include "minillm/core/dtype.h"
#include "minillm/core/device.h"
#include "minillm/core/shape.h"
#include "minillm/core/tensor.h"
#include "minillm/core/tensor_view.h"

// Graph
#include "minillm/graph/op_type.h"
#include "minillm/graph/attribute.h"
#include "minillm/graph/value.h"
#include "minillm/graph/node.h"
#include "minillm/graph/graph.h"
#include "minillm/graph/shape_infer.h"
#include "minillm/graph/graph_builder.h"

// Runtime
#include "minillm/runtime/allocator.h"
#include "minillm/runtime/backend.h"
#include "minillm/runtime/cpu_backend.h"
#include "minillm/runtime/kernel_registry.h"
#include "minillm/runtime/executor.h"
#include "minillm/runtime/runtime_context.h"
#include "minillm/runtime/cpu_kernels.h"
#include "minillm/runtime/cpu_kernel_adapter.h"
#include "minillm/runtime/memory_planner.h"
#include "minillm/runtime/kv_cache.h"
#include "minillm/runtime/sampler.h"

#if defined(MINILLM_ENABLE_CUDA)
#include "minillm/runtime/cuda_backend.h"
#include "minillm/runtime/cuda_kernel_adapter.h"
#include "minillm/runtime/cuda_kernels.h"
#endif

// Model
#include "minillm/model/transformer_graph_builder.h"

// IO
#include "minillm/io/gguf_format.h"
#include "minillm/io/gguf_parser.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/io/bpe_tokenizer.h"
