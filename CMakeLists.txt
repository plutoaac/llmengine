cmake_minimum_required(VERSION 3.22)
project(MiniLLMEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MINILLM_ENABLE_NATIVE_ARCH "Enable host-native CPU code generation when supported" ON)
option(MINILLM_ENABLE_CUDA "Build optional CUDA backend and FP32 CUDA kernels" OFF)

if(MINILLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(MINILLM_ENABLE_NATIVE_ARCH AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" MINILLM_HAS_MARCH_NATIVE)
    if(MINILLM_HAS_MARCH_NATIVE)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()

include(CTest)

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
    src/runtime/memory_planner.cpp
    src/runtime/sampler.cpp
    src/runtime/kv_cache.cpp
    src/runtime/paged_kv_cache.cpp
)
target_include_directories(minillm_runtime PUBLIC include)
target_link_libraries(minillm_runtime PUBLIC minillm_graph)

if(MINILLM_ENABLE_CUDA)
    target_sources(minillm_runtime PRIVATE
        src/runtime/cuda_backend.cpp
        src/runtime/cuda_kernel_adapter.cpp
        src/runtime/cuda_kernels.cu
    )
    target_compile_definitions(minillm_core PUBLIC MINILLM_ENABLE_CUDA=1)
    target_compile_definitions(minillm_runtime PUBLIC MINILLM_ENABLE_CUDA=1)
    target_link_libraries(minillm_core PUBLIC CUDA::cudart)
    target_link_libraries(minillm_runtime PUBLIC CUDA::cudart)
endif()

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

# Example: CPU kernel benchmark
add_executable(benchmark_cpu examples/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE minillm_runtime)

# Tests
function(add_minillm_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE minillm_model)
    if(BUILD_TESTING)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endfunction()

add_minillm_test(test_shape)
add_minillm_test(test_tensor)
add_minillm_test(test_graph)
add_minillm_test(test_graph_builder)
add_minillm_test(test_runtime)
add_minillm_test(test_cpu_kernels)
add_minillm_test(test_memory_planner)
add_minillm_test(test_paged_kv_cache)

add_minillm_test(test_gguf_parser)
target_link_libraries(test_gguf_parser PRIVATE minillm_io)
