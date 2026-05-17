cmake_minimum_required(VERSION 3.22)
project(MiniLLMEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable SIMD instructions (AVX2/FMA on supported hosts)
add_compile_options(-march=native)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core library
add_library(minillm_core
    src/core/status.cpp
    src/core/dtype.cpp
    src/core/device.cpp
    src/core/shape.cpp
    src/core/tensor.cpp
)
target_include_directories(minillm_core PUBLIC include)

# Graph library
add_library(minillm_graph
    src/graph/op_type.cpp
    src/graph/attribute.cpp
    src/graph/value.cpp
    src/graph/node.cpp
    src/graph/graph.cpp
    src/graph/shape_infer.cpp
    src/graph/graph_builder.cpp
)
target_include_directories(minillm_graph PUBLIC include)
target_link_libraries(minillm_graph PUBLIC minillm_core)

# Runtime library
add_library(minillm_runtime
    src/runtime/allocator.cpp
    src/runtime/cpu_backend.cpp
    src/runtime/kernel_registry.cpp
    src/runtime/executor.cpp
    src/runtime/runtime_context.cpp
    src/runtime/cpu_kernels.cpp
    src/runtime/cpu_kernel_adapter.cpp
    src/runtime/sampler.cpp
    src/runtime/kv_cache.cpp
)
target_include_directories(minillm_runtime PUBLIC include)
target_link_libraries(minillm_runtime PUBLIC minillm_graph)

# Model library
add_library(minillm_model
    src/model/transformer_graph_builder.cpp
)
target_include_directories(minillm_model PUBLIC include)
target_link_libraries(minillm_model PUBLIC minillm_runtime)

# IO library (GGUF parser + weight loader + tokenizer)
add_library(minillm_io
    src/io/gguf_format.cpp
    src/io/gguf_parser.cpp
    src/io/gguf_weight_loader.cpp
    src/io/bpe_tokenizer.cpp
)
target_include_directories(minillm_io PUBLIC include)
target_link_libraries(minillm_io PUBLIC minillm_model)

# Example: mock execution (graph dump only)
add_executable(build_tiny_llama_graph examples/build_tiny_llama_graph.cpp)
target_link_libraries(build_tiny_llama_graph PRIVATE minillm_model)

# Example: real forward pass with naive CPU kernels
add_executable(forward_tiny_llama examples/forward_tiny_llama.cpp)
target_link_libraries(forward_tiny_llama PRIVATE minillm_model)

# Example: forward pass with GGUF model weights
add_executable(forward_tiny_llama_gguf examples/forward_tiny_llama_gguf.cpp)
target_link_libraries(forward_tiny_llama_gguf PRIVATE minillm_io)

# Example: text generation with GGUF model
add_executable(generate examples/generate.cpp)
target_link_libraries(generate PRIVATE minillm_io)

# Tests
function(add_minillm_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE minillm_model)
endfunction()

add_minillm_test(test_shape)
add_minillm_test(test_tensor)
add_minillm_test(test_graph)
add_minillm_test(test_graph_builder)

add_minillm_test(test_gguf_parser)
target_link_libraries(test_gguf_parser PRIVATE minillm_io)
