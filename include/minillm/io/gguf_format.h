#pragma once

#include <cstdint>
#include <expected>
#include <string_view>

#include "minillm/core/dtype.h"
#include "minillm/core/status.h"

namespace minillm {

// GGUF metadata value type tags (wire format).
enum class GgufValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGML tensor element data types (wire format).
enum class GgmlDataType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    BF16 = 30,
};

constexpr uint32_t kGgufVersion = 3;
constexpr uint32_t kGgufDefaultAlignment = 32;

// Q8_0 block: fp16 scale (2 bytes) + 32 x int8 (32 bytes) = 34 bytes
constexpr size_t kQ8_0BlockSize = 34;
constexpr size_t kQ8_0BlockElems = 32;

// Per-element size for unquantized types; per-block size for block-quantized types.
size_t ggml_dtype_size(GgmlDataType dt);

// Number of elements per block (1 for unquantized, 32 for Q8_0).
size_t ggml_blck_size(GgmlDataType dt);

std::string_view ggml_dtype_name(GgmlDataType dt);

// Map GGML dtype to engine DType. Block-quantized types map to Float32
// (they are dequantized to F32 at load time).
std::expected<DType, Status> map_ggml_dtype(GgmlDataType dt);

} // namespace minillm
