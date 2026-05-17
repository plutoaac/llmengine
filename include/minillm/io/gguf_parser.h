#pragma once

#include <cstdint>
#include <expected>
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/io/gguf_format.h"

namespace minillm {

// Forward declaration for recursive variant
struct MetadataArray;

// Metadata value: variant covering all GGUF metadata types.
using MetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool,
    std::string,
    MetadataArray>;

// Wrapper for metadata arrays (breaks recursive variant cycle)
struct MetadataArray {
    std::vector<MetadataValue> elements;
};

struct GGUFTensorInfo {
    std::string name;
    GgmlDataType dtype{GgmlDataType::F32};
    std::vector<int64_t> dimensions; // row-major (reversed from GGUF column-major)
    uint64_t offset{0};              // relative to data region start

    size_t num_elements() const;
    size_t bytes() const;
};

struct GGUFFile {
    uint32_t version{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
    uint64_t data_offset{0};
    std::unordered_map<std::string, MetadataValue> metadata;
    std::vector<GGUFTensorInfo> tensor_infos;
};

// Metadata access helpers.
std::expected<MetadataValue, Status> get_metadata(
    const std::unordered_map<std::string, MetadataValue>& meta,
    std::string_view key);

template<typename T>
T get_metadata_or(
    const std::unordered_map<std::string, MetadataValue>& meta,
    std::string_view key,
    T default_val) {
    auto it = meta.find(std::string(key));
    if (it == meta.end()) return default_val;
    if (auto* v = std::get_if<T>(&it->second)) return *v;
    return default_val;
}

std::expected<size_t, Status> get_metadata_array_size(
    const std::unordered_map<std::string, MetadataValue>& meta,
    std::string_view key);

// GGUF binary file parser.
// All methods are static; file is opened, read, and closed per call.
class GGUFParser {
public:
    // Parse a GGUF file header (magic, version, metadata, tensor info).
    // Returns GGUFFile on success. File is closed after parse.
    static std::expected<GGUFFile, Status> parse(const std::string& filename);

    // Read raw tensor data from a GGUF file into a caller-supplied buffer.
    static Status read_tensor_data(
        const std::string& filename,
        uint64_t data_offset,
        uint64_t tensor_offset,
        void* dst,
        size_t byte_count);

private:
    static std::expected<uint8_t, Status> read_u8(std::ifstream& f);
    static std::expected<uint16_t, Status> read_u16_le(std::ifstream& f);
    static std::expected<uint32_t, Status> read_u32_le(std::ifstream& f);
    static std::expected<uint64_t, Status> read_u64_le(std::ifstream& f);
    static std::expected<float, Status> read_f32_le(std::ifstream& f);
    static std::expected<double, Status> read_f64_le(std::ifstream& f);
    static std::expected<std::string, Status> read_string(std::ifstream& f);

    static std::expected<MetadataValue, Status> read_metadata_value(
        std::ifstream& f, GgufValueType type);
    static std::expected<std::unordered_map<std::string, MetadataValue>, Status>
        parse_metadata(std::ifstream& f, uint64_t kv_count);
    static std::expected<std::vector<GGUFTensorInfo>, Status>
        parse_tensor_infos(std::ifstream& f, uint64_t tensor_count);
};

} // namespace minillm
