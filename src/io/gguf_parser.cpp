#include "minillm/io/gguf_parser.h"

#include <cstring>
#include <fstream>
#include <limits>

namespace minillm {

std::expected<size_t, Status> GGUFTensorInfo::num_elements() const {
    size_t n = 1;
    for (auto d : dimensions) {
        if (d <= 0) {
            return std::unexpected(Status::invalid_argument(
                "invalid non-positive dimension for tensor: " + name));
        }
        if (n > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
            return std::unexpected(Status::invalid_argument(
                "numel overflow for tensor: " + name));
        }
        n *= static_cast<size_t>(d);
    }
    return n;
}

std::expected<size_t, Status> GGUFTensorInfo::bytes() const {
    auto ne = num_elements();
    if (!ne) return std::unexpected(ne.error());
    size_t elem_size = ggml_dtype_size(dtype);
    if (*ne > std::numeric_limits<size_t>::max() / elem_size) {
        return std::unexpected(Status::invalid_argument(
            "byte size overflow for tensor: " + name));
    }
    return *ne * elem_size;
}

std::expected<MetadataValue, Status> get_metadata(
    const std::unordered_map<std::string, MetadataValue>& meta,
    std::string_view key) {
    auto it = meta.find(std::string(key));
    if (it == meta.end()) {
        return std::unexpected(Status::not_found(
            "metadata key not found: " + std::string(key)));
    }
    return it->second;
}

std::expected<size_t, Status> get_metadata_array_size(
    const std::unordered_map<std::string, MetadataValue>& meta,
    std::string_view key) {
    auto it = meta.find(std::string(key));
    if (it == meta.end()) {
        return std::unexpected(Status::not_found(
            "metadata key not found: " + std::string(key)));
    }
    auto* arr = std::get_if<MetadataArray>(&it->second);
    if (!arr) {
        return std::unexpected(Status::type_error(
            "metadata key is not an array: " + std::string(key)));
    }
    return arr->elements.size();
}

// --- Binary readers ---

std::expected<uint8_t, Status> GGUFParser::read_u8(std::ifstream& f) {
    int ch = f.get();
    if (f.eof()) return std::unexpected(Status::io_error("unexpected EOF reading uint8"));
    return static_cast<uint8_t>(ch);
}

std::expected<uint16_t, Status> GGUFParser::read_u16_le(std::ifstream& f) {
    uint8_t b[2];
    if (!f.read(reinterpret_cast<char*>(b), 2))
        return std::unexpected(Status::io_error("failed to read uint16"));
    return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

std::expected<uint32_t, Status> GGUFParser::read_u32_le(std::ifstream& f) {
    uint8_t b[4];
    if (!f.read(reinterpret_cast<char*>(b), 4))
        return std::unexpected(Status::io_error("failed to read uint32"));
    return static_cast<uint32_t>(b[0]) |
           (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
}

std::expected<uint64_t, Status> GGUFParser::read_u64_le(std::ifstream& f) {
    uint8_t b[8];
    if (!f.read(reinterpret_cast<char*>(b), 8))
        return std::unexpected(Status::io_error("failed to read uint64"));
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(b[i]) << (i * 8);
    return v;
}

std::expected<float, Status> GGUFParser::read_f32_le(std::ifstream& f) {
    auto raw = read_u32_le(f);
    if (!raw) return std::unexpected(raw.error());
    float fval;
    std::memcpy(&fval, &*raw, sizeof(float));
    return fval;
}

std::expected<double, Status> GGUFParser::read_f64_le(std::ifstream& f) {
    auto raw = read_u64_le(f);
    if (!raw) return std::unexpected(raw.error());
    double dval;
    std::memcpy(&dval, &*raw, sizeof(double));
    return dval;
}

std::expected<std::string, Status> GGUFParser::read_string(std::ifstream& f) {
    auto len = read_u64_le(f);
    if (!len) return std::unexpected(len.error());
    constexpr uint64_t max_string_len = 64 * 1024 * 1024; // 64 MiB safety limit
    if (*len > max_string_len) {
        return std::unexpected(Status::invalid_argument(
            "GGUF string too long: " + std::to_string(*len) + " bytes"));
    }
    if (*len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return std::unexpected(Status::invalid_argument(
            "GGUF string length overflows streamsize: " + std::to_string(*len)));
    }
    std::string str(static_cast<size_t>(*len), '\0');
    if (*len > 0 && !f.read(str.data(), static_cast<std::streamsize>(*len)))
        return std::unexpected(Status::io_error("failed to read string"));
    return str;
}

// --- Metadata parsing ---

std::expected<MetadataValue, Status> GGUFParser::read_metadata_value(
    std::ifstream& f, GgufValueType type) {
    switch (type) {
    case GgufValueType::UINT8:   return read_u8(f);
    case GgufValueType::INT8: {
        auto v = read_u8(f);
        if (!v) return std::unexpected(v.error());
        return static_cast<int8_t>(*v);
    }
    case GgufValueType::UINT16:  return read_u16_le(f);
    case GgufValueType::INT16: {
        auto v = read_u16_le(f);
        if (!v) return std::unexpected(v.error());
        return static_cast<int16_t>(*v);
    }
    case GgufValueType::UINT32:  return read_u32_le(f);
    case GgufValueType::INT32: {
        auto v = read_u32_le(f);
        if (!v) return std::unexpected(v.error());
        return static_cast<int32_t>(*v);
    }
    case GgufValueType::UINT64:  return read_u64_le(f);
    case GgufValueType::INT64: {
        auto v = read_u64_le(f);
        if (!v) return std::unexpected(v.error());
        return static_cast<int64_t>(*v);
    }
    case GgufValueType::FLOAT32: return read_f32_le(f);
    case GgufValueType::FLOAT64: return read_f64_le(f);
    case GgufValueType::BOOL: {
        auto v = read_u8(f);
        if (!v) return std::unexpected(v.error());
        return *v != 0;
    }
    case GgufValueType::STRING:  return read_string(f);
    case GgufValueType::ARRAY: {
        auto elem_type_val = read_u32_le(f);
        if (!elem_type_val) return std::unexpected(elem_type_val.error());
        auto array_len = read_u64_le(f);
        if (!array_len) return std::unexpected(array_len.error());
        constexpr uint64_t max_array_len = 16 * 1024 * 1024; // 16M elements safety limit
        if (*array_len > max_array_len) {
            return std::unexpected(Status::invalid_argument(
                "GGUF metadata array too long: " + std::to_string(*array_len)));
        }
        std::vector<MetadataValue> arr;
        arr.reserve(static_cast<size_t>(*array_len));
        for (uint64_t i = 0; i < *array_len; ++i) {
            auto elem = read_metadata_value(f, static_cast<GgufValueType>(*elem_type_val));
            if (!elem) return std::unexpected(elem.error());
            arr.push_back(std::move(*elem));
        }
        return MetadataArray{std::move(arr)};
    }
    default:
        return std::unexpected(Status::unsupported(
            "unsupported GGUF metadata type: " + std::to_string(static_cast<uint32_t>(type))));
    }
}

std::expected<std::unordered_map<std::string, MetadataValue>, Status>
GGUFParser::parse_metadata(std::ifstream& f, uint64_t kv_count) {
    std::unordered_map<std::string, MetadataValue> meta;
    for (uint64_t i = 0; i < kv_count; ++i) {
        auto key = read_string(f);
        if (!key) return std::unexpected(key.error());
        auto type_val = read_u32_le(f);
        if (!type_val) return std::unexpected(type_val.error());
        auto value = read_metadata_value(f, static_cast<GgufValueType>(*type_val));
        if (!value) return std::unexpected(value.error());
        meta.emplace(std::move(*key), std::move(*value));
    }
    return meta;
}

// --- Tensor info parsing ---

std::expected<std::vector<GGUFTensorInfo>, Status>
GGUFParser::parse_tensor_infos(std::ifstream& f, uint64_t tensor_count) {
    constexpr uint64_t max_tensor_count = 1024 * 1024; // safety limit
    if (tensor_count > max_tensor_count) {
        return std::unexpected(Status::invalid_argument(
            "GGUF tensor count too large: " + std::to_string(tensor_count)));
    }
    std::vector<GGUFTensorInfo> infos;
    infos.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info;
        auto name = read_string(f);
        if (!name) return std::unexpected(name.error());
        info.name = std::move(*name);

        auto dims_val = read_u32_le(f);
        if (!dims_val) return std::unexpected(dims_val.error());
        uint32_t dims = *dims_val;
        constexpr uint32_t max_tensor_dims = 8;
        if (dims == 0 || dims > max_tensor_dims) {
            return std::unexpected(Status::invalid_argument(
                "GGUF tensor has invalid dimension count: " + std::to_string(dims)));
        }

        info.dimensions.resize(dims);
        // GGUF stores dims in column-major order; reverse to row-major
        for (uint32_t d = 0; d < dims; ++d) {
            auto dim = read_u64_le(f);
            if (!dim) return std::unexpected(dim.error());
            if (*dim == 0 ||
                *dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return std::unexpected(Status::invalid_argument(
                    "GGUF tensor has invalid dimension: " + std::to_string(*dim)));
            }
            info.dimensions[dims - d - 1] = static_cast<int64_t>(*dim);
        }

        auto dtype_val = read_u32_le(f);
        if (!dtype_val) return std::unexpected(dtype_val.error());
        info.dtype = static_cast<GgmlDataType>(*dtype_val);

        auto offset = read_u64_le(f);
        if (!offset) return std::unexpected(offset.error());
        info.offset = *offset;

        infos.push_back(std::move(info));
    }
    return infos;
}

// --- Top-level parse ---

std::expected<GGUFFile, Status> GGUFParser::parse(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        return std::unexpected(Status::io_error("cannot open file: " + filename));
    }

    // Validate magic
    char magic[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return std::unexpected(Status::invalid_argument(
            "invalid GGUF file: magic number mismatch"));
    }

    GGUFFile file;

    auto version = read_u32_le(f);
    if (!version) return std::unexpected(version.error());
    file.version = *version;
    if (file.version != kGgufVersion) {
        return std::unexpected(Status::unsupported(
            "unsupported GGUF version " + std::to_string(file.version) +
            " (only v3 is supported)"));
    }

    auto tc = read_u64_le(f);
    if (!tc) return std::unexpected(tc.error());
    file.tensor_count = *tc;

    auto mc = read_u64_le(f);
    if (!mc) return std::unexpected(mc.error());
    file.metadata_kv_count = *mc;

    auto meta = parse_metadata(f, file.metadata_kv_count);
    if (!meta) return std::unexpected(meta.error());
    file.metadata = std::move(*meta);

    auto infos = parse_tensor_infos(f, file.tensor_count);
    if (!infos) return std::unexpected(infos.error());
    file.tensor_infos = std::move(*infos);

    // Compute data_offset: align current position
    uint32_t alignment = kGgufDefaultAlignment;
    auto align_it = file.metadata.find("general.alignment");
    if (align_it != file.metadata.end()) {
        if (auto* v = std::get_if<uint32_t>(&align_it->second)) alignment = *v;
    }

    auto tell_pos = f.tellg();
    if (tell_pos == std::streampos(-1)) {
        return std::unexpected(Status::io_error(
            "failed to determine stream position after parsing GGUF header"));
    }
    uint64_t raw_offset = static_cast<uint64_t>(tell_pos);
    file.data_offset = raw_offset +
        (alignment - raw_offset % alignment) % alignment;

    return file;
}

Status GGUFParser::read_tensor_data(
    const std::string& filename,
    uint64_t data_offset,
    uint64_t tensor_offset,
    void* dst,
    size_t byte_count) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return Status::io_error("cannot open file: " + filename);

    uint64_t abs_offset = data_offset + tensor_offset;
    f.seekg(static_cast<std::streamoff>(abs_offset));
    if (!f) return Status::io_error(
        "seek to offset " + std::to_string(abs_offset) + " failed");

    if (!f.read(static_cast<char*>(dst), static_cast<std::streamsize>(byte_count)))
        return Status::io_error(
            "read of " + std::to_string(byte_count) + " bytes at offset " +
            std::to_string(abs_offset) + " failed");

    return Status::make_ok();
}

} // namespace minillm
