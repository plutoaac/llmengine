cmake_minimum_required(VERSION 3.22)

option(MINILLM_ENABLE_CUDA "Build optional CUDA backend and FP32 CUDA kernels" OFF)

if(MINILLM_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
            "CUDA architectures to build for")
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        find_program(MINILLM_CUDA_HOST_CXX g++-12)
        if(NOT MINILLM_CUDA_HOST_CXX)
            find_program(MINILLM_CUDA_HOST_CXX g++-11)
        endif()
        if(MINILLM_CUDA_HOST_CXX)
            set(CMAKE_CUDA_HOST_COMPILER "${MINILLM_CUDA_HOST_CXX}" CACHE FILEPATH
                "Host compiler used by nvcc")
        endif()
    endif()
endif()

project(MiniLLMEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MINILLM_ENABLE_NATIVE_ARCH "Enable host-native CPU code generation when supported" ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif()

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
    src/core/device.cpp
    src/core/shape.cpp
    src/core/tensor.cpp
)
target_include_directories(minillm_core PUBLIC include)

# Graph library
add_library(minillm_graph
    src/graph/attribute.cpp
    src/graph/node.cpp
    src/graph/graph.cpp
    src/graph/shape_infer.cpp
    src/graph/graph_builder.cpp
)
target_include_directories(minillm_graph PUBLIC include)
target_link_libraries(minillm_graph PUBLIC minillm_core)

# Runtime library
add_library(minillm_runtime
    src/runtime/backend.cpp
    src/runtime/kernel_registry.cpp
    src/runtime/executor.cpp
    src/runtime/runtime_context.cpp
    src/runtime/memory_planner.cpp
    src/runtime/sampler.cpp
    src/runtime/continuous_batch_scheduler.cpp
    # Kernel layer
    src/runtime/kernels/cpu_kernels.cpp
    src/runtime/kernels/cpu_kernel_adapter.cpp
    # Cache layer
    src/runtime/cache/kv_cache.cpp
    src/runtime/cache/paged_kv_cache.cpp
    src/runtime/cache/paged_attention_scheduler.cpp
)
target_include_directories(minillm_runtime PUBLIC include)
target_link_libraries(minillm_runtime PUBLIC minillm_graph)
if(OpenMP_CXX_FOUND)
    target_link_libraries(minillm_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MINILLM_ENABLE_CUDA)
    target_sources(minillm_runtime PRIVATE
        src/runtime/kernels/cuda_kernel_adapter.cpp
        src/runtime/kernels/cuda_kernels.cu
    )
    target_compile_definitions(minillm_core PUBLIC MINILLM_ENABLE_CUDA=1)
    target_compile_definitions(minillm_runtime PUBLIC MINILLM_ENABLE_CUDA=1)
    target_link_libraries(minillm_core PUBLIC CUDA::cudart)
    target_link_libraries(minillm_runtime PUBLIC CUDA::cudart CUDA::cublas)
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

# Examples
add_executable(generate examples/generate.cpp)
target_link_libraries(generate PRIVATE minillm_io)

add_executable(generate_paged examples/generate_paged.cpp)
target_link_libraries(generate_paged PRIVATE minillm_io)

add_executable(benchmark_cpu examples/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE minillm_runtime)

add_executable(benchmark_flash_attention examples/benchmark_flash_attention.cpp)
target_link_libraries(benchmark_flash_attention PRIVATE minillm_runtime)

# Tools
add_executable(dump_tokens tools/dump_tokens.cpp)
target_link_libraries(dump_tokens PRIVATE minillm_io)

if(MINILLM_ENABLE_CUDA)
    add_executable(generate_cuda examples/generate_cuda.cpp)
    target_link_libraries(generate_cuda PRIVATE minillm_io CUDA::cudart)

    add_executable(benchmark_cuda examples/benchmark_cuda.cpp)
    target_link_libraries(benchmark_cuda PRIVATE minillm_runtime CUDA::cudart)

    add_executable(test_cuda_kernels tests/test_cuda_kernels.cpp)
    target_link_libraries(test_cuda_kernels PRIVATE minillm_model CUDA::cudart)
    if(MSVC)
        target_compile_options(test_cuda_kernels PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_cuda_kernels PRIVATE -UNDEBUG)
    endif()
    if(BUILD_TESTING)
        add_test(NAME test_cuda_kernels COMMAND test_cuda_kernels)
    endif()
endif()

# Tests
function(add_minillm_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE minillm_model)
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${test_name} PRIVATE -UNDEBUG)
    endif()
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
add_minillm_test(test_cpu_kernels_bf16)
add_minillm_test(test_memory_planner)
add_minillm_test(test_paged_kv_cache)
add_minillm_test(test_paged_attention_scheduler)
add_minillm_test(test_continuous_batch_scheduler)
add_minillm_test(test_gguf_parser)
target_link_libraries(test_gguf_parser PRIVATE minillm_io)
add_minillm_test(test_transformer_graph_builder)
add_minillm_test(test_bpe_tokenizer)
target_link_libraries(test_bpe_tokenizer PRIVATE minillm_io)
add_minillm_test(test_e2e_verification)
target_link_libraries(test_e2e_verification PRIVATE minillm_io)
