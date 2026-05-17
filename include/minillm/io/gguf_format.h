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
// Only F32/F16/BF16 are supported initially; others will fail at load time.
enum class GgmlDataType : uint32_t {
    F32 = 0,
    F16 = 1,
    // Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
    // Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
    // I8=24, I16=25, I32=26, I64=27, F64=28,
    BF16 = 30,
};

constexpr uint32_t kGgufVersion = 3;
constexpr uint32_t kGgufDefaultAlignment = 32;

size_t ggml_dtype_size(GgmlDataType dt);
std::string_view ggml_dtype_name(GgmlDataType dt);

std::expected<DType, Status> map_ggml_dtype(GgmlDataType dt);

} // namespace minillm
