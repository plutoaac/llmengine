#include "minillm/io/gguf_parser.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace minillm {

class GGUFMemoryView {
public:
    virtual ~GGUFMemoryView() = default;
    virtual std::span<const std::byte> bytes() const noexcept = 0;
};

namespace {

#if defined(_WIN32)
std::string win32_message(const std::string& what) {
    return what + ": GetLastError=" + std::to_string(static_cast<unsigned long>(GetLastError()));
}

class WindowsMappedGGUFMemoryView final : public GGUFMemoryView {
public:
    WindowsMappedGGUFMemoryView(HANDLE file, HANDLE mapping, void* view, size_t size)
        : file_(file), mapping_(mapping), view_(view), size_(size) {}

    ~WindowsMappedGGUFMemoryView() override {
        if (view_) UnmapViewOfFile(view_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

    std::span<const std::byte> bytes() const noexcept override {
        return std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(view_), size_);
    }

    static std::expected<std::shared_ptr<const GGUFMemoryView>, Status> open(
        const std::string& filename) {
        HANDLE file = CreateFileA(
            filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return std::unexpected(Status::io_error(
                win32_message("failed to open GGUF file for mapping: " + filename)));
        }

        LARGE_INTEGER size_li;
        if (!GetFileSizeEx(file, &size_li)) {
            CloseHandle(file);
            return std::unexpected(Status::io_error(
                win32_message("failed to query GGUF file size: " + filename)));
        }
        if (size_li.QuadPart < 0 ||
            static_cast<unsigned long long>(size_li.QuadPart) >
                static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            CloseHandle(file);
            return std::unexpected(Status::invalid_argument(
                "GGUF file too large to map: " + filename));
        }

        const size_t size = static_cast<size_t>(size_li.QuadPart);
        if (size == 0) {
            CloseHandle(file);
            return std::unexpected(Status::invalid_argument(
                "cannot mmap empty GGUF file: " + filename));
        }

        HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            CloseHandle(file);
            return std::unexpected(Status::io_error(
                win32_message("failed to create file mapping: " + filename)));
        }

        void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            CloseHandle(mapping);
            CloseHandle(file);
            return std::unexpected(Status::io_error(
                win32_message("failed to map GGUF file: " + filename)));
        }

        return std::shared_ptr<const GGUFMemoryView>(
            new WindowsMappedGGUFMemoryView(file, mapping, view, size));
    }

private:
    HANDLE file_{INVALID_HANDLE_VALUE};
    HANDLE mapping_{nullptr};
    void* view_{nullptr};
    size_t size_{0};
};
#else
std::string errno_message(const std::string& what) {
    return what + ": " + std::strerror(errno);
}

class PosixMappedGGUFMemoryView final : public GGUFMemoryView {
public:
    PosixMappedGGUFMemoryView(int fd, void* mapping, size_t size)
        : fd_(fd), mapping_(mapping), size_(size) {}

    ~PosixMappedGGUFMemoryView() override {
        if (mapping_ != MAP_FAILED) munmap(mapping_, size_);
        if (fd_ >= 0) close(fd_);
    }

    std::span<const std::byte> bytes() const noexcept override {
        return std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(mapping_), size_);
    }

    static std::expected<std::shared_ptr<const GGUFMemoryView>, Status> open(
        const std::string& filename) {
        int fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0) {
            return std::unexpected(Status::io_error(
                errno_message("failed to open GGUF file for mapping: " + filename)));
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            int saved_errno = errno;
            close(fd);
            errno = saved_errno;
            return std::unexpected(Status::io_error(
                errno_message("failed to query GGUF file size: " + filename)));
        }
        if (st.st_size <= 0 ||
            static_cast<unsigned long long>(st.st_size) >
                static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            close(fd);
            return std::unexpected(Status::invalid_argument(
                "GGUF file too large to map: " + filename));
        }

        const size_t size = static_cast<size_t>(st.st_size);
        void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            int saved_errno = errno;
            close(fd);
            errno = saved_errno;
            return std::unexpected(Status::io_error(
                errno_message("failed to mmap GGUF file: " + filename)));
        }

        return std::shared_ptr<const GGUFMemoryView>(
            new PosixMappedGGUFMemoryView(fd, mapping, size));
    }

private:
    int fd_{-1};
    void* mapping_{MAP_FAILED};
    size_t size_{0};
};
#endif

std::expected<std::shared_ptr<const GGUFMemoryView>, Status> try_open_memory_view(
    const std::string& filename) {
#if defined(_WIN32)
    return WindowsMappedGGUFMemoryView::open(filename);
#else
    return PosixMappedGGUFMemoryView::open(filename);
#endif
}

Status copy_tensor_bytes_from_view(
    const std::span<const std::byte> bytes,
    uint64_t data_offset,
    uint64_t tensor_offset,
    void* dst,
    size_t byte_count) {
    if (data_offset > std::numeric_limits<uint64_t>::max() - tensor_offset) {
        return Status::out_of_range("tensor byte offset overflow");
    }
    uint64_t abs_offset = data_offset + tensor_offset;
    uint64_t view_size = static_cast<uint64_t>(bytes.size());
    if (abs_offset > view_size || byte_count > view_size - abs_offset) {
        return Status::out_of_range(
            "tensor byte range exceeds mapped GGUF file bounds");
    }
    std::memcpy(dst, bytes.data() + static_cast<size_t>(abs_offset), byte_count);
    return Status::make_ok();
}

} // namespace

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
    size_t blck = ggml_blck_size(dtype);
    size_t dtype_sz = ggml_dtype_size(dtype);
    if (blck == 0 || dtype_sz == 0) {
        return std::unexpected(Status::unsupported(
            "unsupported GGML dtype for tensor byte size: " + name));
    }
    // For block-quantized types: (numel / blck_size) * dtype_size
    // For unquantized types (blck_size=1): numel * dtype_size
    size_t num_blocks = (*ne + blck - 1) / blck;
    if (num_blocks > std::numeric_limits<size_t>::max() / dtype_sz) {
        return std::unexpected(Status::invalid_argument(
            "byte size overflow for tensor: " + name));
    }
    return num_blocks * dtype_sz;
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

std::expected<std::span<const std::byte>, Status> GGUFFile::tensor_view(
    const GGUFTensorInfo& ti) const {
    if (!memory_view) {
        return std::unexpected(Status::unsupported(
            "GGUF memory view unavailable for file: " +
            (source_path.empty() ? std::string("<unknown>") : source_path)));
    }

    auto raw = memory_view->bytes();
    auto byte_count_exp = ti.bytes();
    if (!byte_count_exp) return std::unexpected(byte_count_exp.error());

    if (data_offset > std::numeric_limits<uint64_t>::max() - ti.offset) {
        return std::unexpected(Status::out_of_range(
            "tensor offset overflow for: " + ti.name));
    }
    uint64_t abs_offset = data_offset + ti.offset;
    if (abs_offset > raw.size() ||
        *byte_count_exp > raw.size() - static_cast<size_t>(abs_offset)) {
        return std::unexpected(Status::out_of_range(
            "tensor byte range exceeds mapped GGUF file bounds for: " + ti.name));
    }

    return raw.subspan(static_cast<size_t>(abs_offset), *byte_count_exp);
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
    file.source_path = filename;

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

    auto memory_view = try_open_memory_view(filename);
    if (memory_view) {
        file.memory_view = std::move(*memory_view);
    }

    return file;
}

Status GGUFParser::read_tensor_data(
    const std::string& filename,
    uint64_t data_offset,
    uint64_t tensor_offset,
    void* dst,
    size_t byte_count) {
    auto memory_view = try_open_memory_view(filename);
    if (memory_view) {
        auto bytes = (*memory_view)->bytes();
        auto st = copy_tensor_bytes_from_view(
            bytes, data_offset, tensor_offset, dst, byte_count);
        if (!st.ok()) return st;
        return Status::make_ok();
    }

    std::ifstream f(filename, std::ios::binary);
    if (!f) return Status::io_error("cannot open file: " + filename);

    if (data_offset > std::numeric_limits<uint64_t>::max() - tensor_offset) {
        return Status::out_of_range("tensor offset overflow for: " + filename);
    }
    uint64_t abs_offset = data_offset + tensor_offset;
    if (abs_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return Status::out_of_range(
            "tensor offset exceeds stream seek range for: " + filename);
    }
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
