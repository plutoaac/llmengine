#include "minillm/io/gguf_format.h"

namespace minillm {

size_t ggml_dtype_size(GgmlDataType dt) {
    switch (dt) {
    case GgmlDataType::F32: return 4;
    case GgmlDataType::F16: return 2;
    case GgmlDataType::BF16: return 2;
    case GgmlDataType::Q8_0: return kQ8_0BlockSize;
    default: return 0;
    }
}

size_t ggml_blck_size(GgmlDataType dt) {
    switch (dt) {
    case GgmlDataType::F32:
    case GgmlDataType::F16:
    case GgmlDataType::BF16:
        return 1;
    case GgmlDataType::Q8_0:
        return kQ8_0BlockElems;
    default:
        return 0;
    }
}

std::string_view ggml_dtype_name(GgmlDataType dt) {
    switch (dt) {
    case GgmlDataType::F32:  return "F32";
    case GgmlDataType::F16:  return "F16";
    case GgmlDataType::BF16: return "BF16";
    case GgmlDataType::Q8_0: return "Q8_0";
    default: return "UNKNOWN";
    }
}

std::expected<DType, Status> map_ggml_dtype(GgmlDataType dt) {
    switch (dt) {
    case GgmlDataType::F32:  return DType::Float32;
    case GgmlDataType::F16:  return DType::Float16;
    case GgmlDataType::BF16: return DType::BFloat16;
    case GgmlDataType::Q8_0: return DType::Float32;
    default:
        return std::unexpected(Status::unsupported(
            std::string("unsupported GGML dtype ") +
            std::string(ggml_dtype_name(dt)) +
            " (only F32/F16/BF16/Q8_0 are supported)"));
    }
}

} // namespace minillm
