#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "minillm/io/gguf_format.h"
#include "minillm/io/gguf_parser.h"
#include "minillm/io/gguf_weight_loader.h"

using namespace minillm;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        try { test_##name(); \
            std::cout << "  PASS test_" #name "\n"; \
            ++tests_passed; \
        } catch (const std::exception& e) { \
            std::cout << "  FAIL test_" #name ": " << e.what() << "\n"; \
            ++tests_failed; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) throw std::runtime_error( \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
        ": assertion failed"); } while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) throw std::runtime_error( \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
        ": assertion failed: " #cond); } while(0)

// --- GGUF test file writer ---

class GgufWriter {
public:
    explicit GgufWriter(std::string path) : path_(std::move(path)) {}

    void write_u8(uint8_t v) { buf_.push_back(v); }
    void write_u16_le(uint16_t v) {
        buf_.push_back(v & 0xFF);
        buf_.push_back((v >> 8) & 0xFF);
    }
    void write_u32_le(uint32_t v) {
        for (int i = 0; i < 4; ++i) buf_.push_back((v >> (i * 8)) & 0xFF);
    }
    void write_u64_le(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf_.push_back((v >> (i * 8)) & 0xFF);
    }
    void write_string(const std::string& s) {
        write_u64_le(s.size());
        for (char c : s) buf_.push_back(static_cast<uint8_t>(c));
    }
    void write_float_le(float v) {
        uint32_t raw;
        std::memcpy(&raw, &v, 4);
        write_u32_le(raw);
    }

    // Add metadata KV pair (string value)
    void add_meta_string(const std::string& key, const std::string& val) {
        meta_keys_.push_back(key);
        meta_values_.push_back(val);
    }
    // Add metadata KV pair (uint32 value)
    void add_meta_u32(const std::string& key, uint32_t val) {
        meta_u32_keys_.push_back(key);
        meta_u32_values_.push_back(val);
    }

    // Add a 2D F32 tensor
    void add_tensor_f32(const std::string& name, int64_t d0, int64_t d1,
                        const std::vector<float>& data) {
        TensorEntry e;
        e.name = name;
        e.dims = {d0, d1};
        e.dtype = static_cast<uint32_t>(GgmlDataType::F32);
        e.data_f32 = data;
        tensors_.push_back(std::move(e));
    }

    // Add a 2D F16 tensor
    void add_tensor_f16(const std::string& name, int64_t d0, int64_t d1,
                        const std::vector<uint16_t>& data) {
        TensorEntry e;
        e.name = name;
        e.dims = {d0, d1};
        e.dtype = static_cast<uint32_t>(GgmlDataType::F16);
        e.data_u16 = data;
        tensors_.push_back(std::move(e));
    }

    // Add a 2D BF16 tensor
    void add_tensor_bf16(const std::string& name, int64_t d0, int64_t d1,
                         const std::vector<uint16_t>& data) {
        TensorEntry e;
        e.name = name;
        e.dims = {d0, d1};
        e.dtype = static_cast<uint32_t>(GgmlDataType::BF16);
        e.data_u16 = data;
        tensors_.push_back(std::move(e));
    }

    void write() {
        buf_.clear();

        // Magic
        buf_.push_back('G'); buf_.push_back('G');
        buf_.push_back('U'); buf_.push_back('F');

        // Version
        write_u32_le(3);

        // Tensor count
        write_u64_le(tensors_.size());

        // Metadata KV count
        uint64_t kv_count = meta_keys_.size() + meta_u32_keys_.size();
        write_u64_le(kv_count);

        // Metadata string entries
        for (size_t i = 0; i < meta_keys_.size(); ++i) {
            write_string(meta_keys_[i]);
            write_u32_le(static_cast<uint32_t>(GgufValueType::STRING)); // type
            write_string(meta_values_[i]);
        }
        // Metadata uint32 entries
        for (size_t i = 0; i < meta_u32_keys_.size(); ++i) {
            write_string(meta_u32_keys_[i]);
            write_u32_le(static_cast<uint32_t>(GgufValueType::UINT32)); // type
            write_u32_le(meta_u32_values_[i]);
        }

        // Tensor info entries
        for (auto& t : tensors_) {
            write_string(t.name);
            write_u32_le(static_cast<uint32_t>(t.dims.size()));
            // GGUF stores dims in column-major, so write reversed
            for (size_t d = t.dims.size(); d > 0; --d) {
                write_u64_le(static_cast<uint64_t>(t.dims[d - 1]));
            }
            write_u32_le(t.dtype);
            write_u64_le(t.data_offset); // placeholder, fixed below
        }

        // Pad to alignment
        uint64_t raw_offset = buf_.size();
        uint64_t alignment = 32;
        uint64_t data_offset = raw_offset +
            (alignment - raw_offset % alignment) % alignment;
        while (buf_.size() < data_offset) buf_.push_back(0);

        // Write tensor data, fixing up offsets
        size_t current_offset = 0;
        for (auto& t : tensors_) {
            // Fix the offset in the header
            // Find where the offset was written in the buffer
            // (We'll just rewrite the whole thing with correct offsets)
            t.data_offset = current_offset;

            size_t elem_count = 1;
            for (auto d : t.dims) elem_count *= static_cast<size_t>(d);

            if (t.dtype == static_cast<uint32_t>(GgmlDataType::F32)) {
                for (size_t i = 0; i < elem_count; ++i) {
                    float val = (i < t.data_f32.size()) ? t.data_f32[i] : 0.0f;
                    write_float_le(val);
                }
                current_offset += elem_count * 4;
            } else {
                // F16 or BF16
                for (size_t i = 0; i < elem_count; ++i) {
                    uint16_t val = (i < t.data_u16.size()) ? t.data_u16[i] : 0;
                    write_u16_le(val);
                }
                current_offset += elem_count * 2;
            }
        }

        // Now rewrite the whole file with correct tensor offsets
        // (easier to do a second pass)
        rewrite_with_offsets(data_offset);

        std::ofstream f(path_, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf_.data()),
                static_cast<std::streamsize>(buf_.size()));
    }

private:
    struct TensorEntry {
        std::string name;
        std::vector<int64_t> dims;
        uint32_t dtype;
        uint64_t data_offset = 0;
        std::vector<float> data_f32;
        std::vector<uint16_t> data_u16;
    };

    void rewrite_with_offsets(uint64_t data_offset_start) {
        // Recompute the full buffer with correct offsets
        std::vector<uint8_t> new_buf;

        // Copy everything up to tensor info entries
        // We'll regenerate from scratch
        new_buf.push_back('G'); new_buf.push_back('G');
        new_buf.push_back('U'); new_buf.push_back('F');
        // version
        for (int i = 0; i < 4; ++i)
            new_buf.push_back((uint32_t{3} >> (i * 8)) & 0xFF);
        // tensor count
        {
            uint64_t v = tensors_.size();
            for (int i = 0; i < 8; ++i) new_buf.push_back((v >> (i * 8)) & 0xFF);
        }
        // kv count
        {
            uint64_t v = meta_keys_.size() + meta_u32_keys_.size();
            for (int i = 0; i < 8; ++i) new_buf.push_back((v >> (i * 8)) & 0xFF);
        }

        // Reuse helper: swap buf_ with new_buf temporarily
        // Actually, let's just rebuild in a simpler way
        buf_ = std::move(new_buf);

        // Metadata string entries
        for (size_t i = 0; i < meta_keys_.size(); ++i) {
            write_string(meta_keys_[i]);
            write_u32_le(static_cast<uint32_t>(GgufValueType::STRING));
            write_string(meta_values_[i]);
        }
        for (size_t i = 0; i < meta_u32_keys_.size(); ++i) {
            write_string(meta_u32_keys_[i]);
            write_u32_le(static_cast<uint32_t>(GgufValueType::UINT32));
            write_u32_le(meta_u32_values_[i]);
        }

        // Compute tensor data offsets
        uint64_t current_data_offset = 0;
        for (auto& t : tensors_) {
            t.data_offset = current_data_offset;
            size_t elem_count = 1;
            for (auto d : t.dims) elem_count *= static_cast<size_t>(d);
            if (t.dtype == static_cast<uint32_t>(GgmlDataType::F32)) {
                current_data_offset += elem_count * 4;
            } else {
                current_data_offset += elem_count * 2;
            }
        }

        // Tensor info entries
        for (auto& t : tensors_) {
            write_string(t.name);
            write_u32_le(static_cast<uint32_t>(t.dims.size()));
            for (size_t d = t.dims.size(); d > 0; --d) {
                write_u64_le(static_cast<uint64_t>(t.dims[d - 1]));
            }
            write_u32_le(t.dtype);
            write_u64_le(t.data_offset);
        }

        // Pad to alignment
        uint64_t raw_offset = buf_.size();
        while (buf_.size() < data_offset_start) buf_.push_back(0);

        // Tensor data
        for (auto& t : tensors_) {
            size_t elem_count = 1;
            for (auto d : t.dims) elem_count *= static_cast<size_t>(d);
            if (t.dtype == static_cast<uint32_t>(GgmlDataType::F32)) {
                for (size_t i = 0; i < elem_count; ++i) {
                    float val = (i < t.data_f32.size()) ? t.data_f32[i] : 0.0f;
                    write_float_le(val);
                }
            } else {
                for (size_t i = 0; i < elem_count; ++i) {
                    uint16_t val = (i < t.data_u16.size()) ? t.data_u16[i] : 0;
                    write_u16_le(val);
                }
            }
        }
    }

    std::string path_;
    std::vector<uint8_t> buf_;
    std::vector<std::string> meta_keys_, meta_values_;
    std::vector<std::string> meta_u32_keys_;
    std::vector<uint32_t> meta_u32_values_;
    std::vector<TensorEntry> tensors_;
};

// --- Tests ---

void test_ggml_dtype_size() {
    ASSERT_EQ(ggml_dtype_size(GgmlDataType::F32), size_t(4));
    ASSERT_EQ(ggml_dtype_size(GgmlDataType::F16), size_t(2));
    ASSERT_EQ(ggml_dtype_size(GgmlDataType::BF16), size_t(2));
}

void test_map_ggml_dtype() {
    auto f32 = map_ggml_dtype(GgmlDataType::F32);
    ASSERT_TRUE(f32.has_value());
    ASSERT_TRUE(*f32 == DType::Float32);

    auto f16 = map_ggml_dtype(GgmlDataType::F16);
    ASSERT_TRUE(f16.has_value());
    ASSERT_TRUE(*f16 == DType::Float16);

    auto bf16 = map_ggml_dtype(GgmlDataType::BF16);
    ASSERT_TRUE(bf16.has_value());
    ASSERT_TRUE(*bf16 == DType::BFloat16);

    // Unsupported type should fail
    auto bad = map_ggml_dtype(static_cast<GgmlDataType>(99));
    ASSERT_TRUE(!bad.has_value());
}

void test_metadata_variant() {
    MetadataValue v_uint32 = uint32_t(42);
    ASSERT_TRUE(std::get_if<uint32_t>(&v_uint32) != nullptr);
    ASSERT_EQ(std::get<uint32_t>(v_uint32), uint32_t(42));

    MetadataValue v_str = std::string("hello");
    ASSERT_TRUE(std::get_if<std::string>(&v_str) != nullptr);
    ASSERT_EQ(std::get<std::string>(v_str), "hello");

    MetadataValue v_arr = MetadataArray{{uint32_t(1), uint32_t(2)}};
    auto* arr = std::get_if<MetadataArray>(&v_arr);
    ASSERT_TRUE(arr != nullptr);
    ASSERT_EQ(arr->elements.size(), size_t(2));
}

void test_get_metadata_or() {
    std::unordered_map<std::string, MetadataValue> meta;
    meta["answer"] = uint32_t(42);
    meta["name"] = std::string("test");

    ASSERT_EQ(get_metadata_or<uint32_t>(meta, "answer", uint32_t(0)), uint32_t(42));
    ASSERT_EQ(get_metadata_or<uint32_t>(meta, "missing", uint32_t(99)), uint32_t(99));
    ASSERT_EQ(get_metadata_or<std::string>(meta, "name", std::string("")), std::string("test"));
    // Wrong type returns default
    ASSERT_EQ(get_metadata_or<uint32_t>(meta, "name", uint32_t(0)), uint32_t(0));
}

void test_parse_minimal_gguf() {
    // Create a minimal GGUF file with one F32 tensor [2, 3]
    std::string path = "/tmp/test_minillm_gguf_minimal.gguf";
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_meta_u32("llama.embedding_length", 32);
        w.add_tensor_f32("test.weight", 2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        w.write();
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(result.has_value());

    auto& file = *result;
    ASSERT_EQ(file.version, uint32_t(3));
    ASSERT_EQ(file.tensor_count, uint64_t(1));
    ASSERT_EQ(file.tensor_infos.size(), size_t(1));
    ASSERT_EQ(file.tensor_infos[0].name, "test.weight");
    ASSERT_EQ(file.tensor_infos[0].dtype, GgmlDataType::F32);
    // Dimensions should be reversed: GGUF stores [3,2], we get [2,3]
    ASSERT_EQ(file.tensor_infos[0].dimensions.size(), size_t(2));
    ASSERT_EQ(file.tensor_infos[0].dimensions[0], int64_t(2));
    ASSERT_EQ(file.tensor_infos[0].dimensions[1], int64_t(3));

    // Check metadata
    auto arch = get_metadata_or<std::string>(file.metadata, "general.architecture", "");
    ASSERT_EQ(arch, "llama");
}

void test_parse_metadata_types() {
    std::string path = "/tmp/test_minillm_gguf_meta.gguf";
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "qwen3");
        w.add_meta_u32("qwen3.embedding_length", 64);
        w.add_meta_u32("qwen3.attention.head_count", 8);
        w.add_tensor_f32("dummy", 1, 1, {0.0f});
        w.write();
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(result.has_value());

    auto& meta = result->metadata;
    ASSERT_EQ(get_metadata_or<std::string>(meta, "general.architecture", ""), "qwen3");
    ASSERT_EQ(get_metadata_or<uint32_t>(meta, "qwen3.embedding_length", uint32_t(0)), uint32_t(64));
    ASSERT_EQ(get_metadata_or<uint32_t>(meta, "qwen3.attention.head_count", uint32_t(0)), uint32_t(8));
}

void test_read_tensor_data() {
    std::string path = "/tmp/test_minillm_gguf_read.gguf";
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_tensor_f32("data.weight", 2, 3, expected);
        w.write();
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(result.has_value());

    auto& ti = result->tensor_infos[0];
    ASSERT_EQ(ti.num_elements(), size_t(6));
    ASSERT_EQ(ti.bytes(), size_t(24));

    std::vector<float> read_buf(6);
    auto st = GGUFParser::read_tensor_data(
        path, result->data_offset, ti.offset,
        read_buf.data(), ti.bytes());
    ASSERT_TRUE(st.ok());

    for (size_t i = 0; i < 6; ++i) {
        ASSERT_TRUE(std::abs(read_buf[i] - expected[i]) < 1e-6f);
    }
}

void test_invalid_magic() {
    std::string path = "/tmp/test_minillm_gguf_bad_magic.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("BADM", 4);
        uint32_t version = 3;
        f.write(reinterpret_cast<const char*>(&version), 4);
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(!result.has_value());
}

void test_name_mapping() {
    // Layer-prefixed names
    auto r1 = WeightLoader::gguf_name_to_value_name("blk.0.attn_q.weight", 0);
    ASSERT_TRUE(r1.has_value());
    ASSERT_EQ(*r1, "layer_0.q_proj.weight");

    auto r1b = WeightLoader::gguf_name_to_value_name("blk.5.attn_q.weight", 5);
    ASSERT_TRUE(r1b.has_value());
    ASSERT_EQ(*r1b, "layer_5.q_proj.weight");

    auto r2 = WeightLoader::gguf_name_to_value_name("blk.0.ffn_gate.weight", 0);
    ASSERT_TRUE(r2.has_value());
    ASSERT_EQ(*r2, "layer_0.gate_proj.weight");

    // QK-norm names
    auto r_qn = WeightLoader::gguf_name_to_value_name("blk.3.attn_q_norm.weight", 3);
    ASSERT_TRUE(r_qn.has_value());
    ASSERT_EQ(*r_qn, "layer_3.q_norm.weight");

    auto r_kn = WeightLoader::gguf_name_to_value_name("blk.3.attn_k_norm.weight", 3);
    ASSERT_TRUE(r_kn.has_value());
    ASSERT_EQ(*r_kn, "layer_3.k_norm.weight");

    // Non-layer names
    auto r3 = WeightLoader::gguf_name_to_value_name("token_embd.weight");
    ASSERT_TRUE(r3.has_value());
    ASSERT_EQ(*r3, "tok_embeddings.weight");

    auto r4 = WeightLoader::gguf_name_to_value_name("output.weight");
    ASSERT_TRUE(r4.has_value());
    ASSERT_EQ(*r4, "lm_head.weight");

    auto r5 = WeightLoader::gguf_name_to_value_name("output_norm.weight");
    ASSERT_TRUE(r5.has_value());
    ASSERT_EQ(*r5, "output_norm.weight");

    // Unknown name
    auto r6 = WeightLoader::gguf_name_to_value_name("unknown.weight");
    ASSERT_TRUE(!r6.has_value());
}

void test_f16_to_f32() {
    // Test zero
    uint16_t f16_zero = 0;
    // Access via dequantize_to_f32
    std::vector<uint16_t> f16_data = {0x0000, // +0.0
                                       0x3C00, // 1.0
                                       0x4000, // 2.0
                                       0x8000, // -0.0
                                       0xBC00, // -1.0
                                       0x7BFF, // max subnormal ~6.1e-5
                                       0x7C00, // +inf
                                       0xFC00}; // -inf
    std::vector<float> f32_out(f16_data.size());

    WeightLoader::dequantize_to_f32(GgmlDataType::F16,
        f16_data.data(), f32_out.data(), f16_data.size());

    ASSERT_TRUE(std::abs(f32_out[0]) < 1e-10f); // +0.0
    ASSERT_TRUE(std::abs(f32_out[1] - 1.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[2] - 2.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[3]) < 1e-10f); // -0.0
    ASSERT_TRUE(std::abs(f32_out[4] + 1.0f) < 1e-6f);
    ASSERT_TRUE(f32_out[5] > 0.0f); // subnormal
    ASSERT_TRUE(std::isinf(f32_out[6])); // +inf
    ASSERT_TRUE(std::isinf(f32_out[7]) && f32_out[7] < 0); // -inf
}

void test_bf16_to_f32() {
    // BF16 is just the top 16 bits of F32
    uint16_t bf16_one = 0x3F80;   // 1.0
    uint16_t bf16_two = 0x4000;   // 2.0
    uint16_t bf16_half = 0x3F00;  // 0.5
    std::vector<uint16_t> bf16_data = {0x0000, bf16_one, bf16_two, bf16_half};
    std::vector<float> f32_out(bf16_data.size());

    WeightLoader::dequantize_to_f32(GgmlDataType::BF16,
        bf16_data.data(), f32_out.data(), bf16_data.size());

    ASSERT_TRUE(std::abs(f32_out[0]) < 1e-10f); // 0.0
    ASSERT_TRUE(std::abs(f32_out[1] - 1.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[2] - 2.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[3] - 0.5f) < 1e-6f);
}

void test_f16_tensor_read() {
    std::string path = "/tmp/test_minillm_gguf_f16.gguf";
    // F16 values: 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200
    std::vector<uint16_t> f16_vals = {0x3C00, 0x4000, 0x4200, 0x4400};
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_tensor_f16("test.weight", 2, 2, f16_vals);
        w.write();
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->tensor_infos[0].dtype, GgmlDataType::F16);

    // Read raw data and dequantize
    auto& ti = result->tensor_infos[0];
    size_t raw_bytes = ti.bytes();
    std::vector<std::byte> raw_buf(raw_bytes);
    auto st = GGUFParser::read_tensor_data(
        path, result->data_offset, ti.offset, raw_buf.data(), raw_bytes);
    ASSERT_TRUE(st.ok());

    std::vector<float> f32_out(4);
    st = WeightLoader::dequantize_to_f32(GgmlDataType::F16,
        raw_buf.data(), f32_out.data(), 4);
    ASSERT_TRUE(st.ok());

    ASSERT_TRUE(std::abs(f32_out[0] - 1.0f) < 1e-4f);
    ASSERT_TRUE(std::abs(f32_out[1] - 2.0f) < 1e-4f);
    ASSERT_TRUE(std::abs(f32_out[2] - 3.0f) < 1e-4f);
    ASSERT_TRUE(std::abs(f32_out[3] - 4.0f) < 1e-4f);
}

void test_bf16_tensor_read() {
    std::string path = "/tmp/test_minillm_gguf_bf16.gguf";
    // BF16: 1.0 = 0x3F80, 2.0 = 0x4000, 0.5 = 0x3F00
    std::vector<uint16_t> bf16_vals = {0x3F80, 0x4000, 0x3F00, 0x4040};
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_tensor_bf16("test.weight", 2, 2, bf16_vals);
        w.write();
    }

    auto result = GGUFParser::parse(path);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->tensor_infos[0].dtype, GgmlDataType::BF16);

    auto& ti = result->tensor_infos[0];
    std::vector<std::byte> raw_buf(ti.bytes());
    auto st = GGUFParser::read_tensor_data(
        path, result->data_offset, ti.offset, raw_buf.data(), ti.bytes());
    ASSERT_TRUE(st.ok());

    std::vector<float> f32_out(4);
    st = WeightLoader::dequantize_to_f32(GgmlDataType::BF16,
        raw_buf.data(), f32_out.data(), 4);
    ASSERT_TRUE(st.ok());

    ASSERT_TRUE(std::abs(f32_out[0] - 1.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[1] - 2.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[2] - 0.5f) < 1e-6f);
    ASSERT_TRUE(std::abs(f32_out[3] - 3.0f) < 1e-6f);
}

void test_extract_config() {
    std::string path = "/tmp/test_minillm_gguf_config.gguf";
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_meta_u32("llama.embedding_length", 64);
        w.add_meta_u32("llama.feed_forward_length", 128);
        w.add_meta_u32("llama.attention.head_count", 8);
        w.add_meta_u32("llama.attention.head_count_kv", 4);
        w.add_meta_u32("llama.attention.key_length", 8);
        w.add_tensor_f32("token_embd.weight", 64, 64, {});
        w.write();
    }

    WeightLoader loader(path);
    auto file = loader.open();
    ASSERT_TRUE(file.has_value());

    auto cfg = loader.extract_config(*file);
    ASSERT_TRUE(cfg.has_value());
    ASSERT_EQ(cfg->hidden_size, int64_t(64));
    ASSERT_EQ(cfg->intermediate_size, int64_t(128));
    ASSERT_EQ(cfg->num_heads, int64_t(8));
    ASSERT_EQ(cfg->num_kv_heads, int64_t(4));
    ASSERT_EQ(cfg->head_dim, int64_t(8));
}

void test_shared_weight_store_tied_embeddings() {
    std::string path = "/tmp/test_minillm_gguf_shared_weights.gguf";
    {
        GgufWriter w(path);
        w.add_meta_string("general.architecture", "llama");
        w.add_tensor_f32("token_embd.weight", 2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        w.write();
    }

    WeightLoader loader(path);
    auto file = loader.open();
    ASSERT_TRUE(file.has_value());

    Graph graph;
    auto tok = graph.add_value("tok_embeddings.weight", Shape({2, 2}),
                               DType::Float32, Device::cpu(), ValueKind::Constant);
    ASSERT_TRUE(tok.has_value());
    auto lm = graph.add_value("lm_head.weight", Shape({2, 2}),
                              DType::Float32, Device::cpu(), ValueKind::Constant);
    ASSERT_TRUE(lm.has_value());

    auto store = loader.load_shared_weights(*file, graph);
    ASSERT_TRUE(store.has_value());
    ASSERT_EQ(store->tensor_count(), size_t(1));
    ASSERT_EQ(store->alias_count(), size_t(2));

    RuntimeContext ctx_a;
    RuntimeContext ctx_b;
    auto st = store->bind(graph, ctx_a);
    ASSERT_TRUE(st.ok());
    st = store->bind(graph, ctx_b);
    ASSERT_TRUE(st.ok());

    Tensor* tok_a = ctx_a.get(*tok);
    Tensor* lm_a = ctx_a.get(*lm);
    Tensor* tok_b = ctx_b.get(*tok);
    ASSERT_TRUE(tok_a != nullptr);
    ASSERT_TRUE(lm_a != nullptr);
    ASSERT_TRUE(tok_b != nullptr);
    ASSERT_TRUE(tok_a == lm_a);
    ASSERT_TRUE(tok_a == tok_b);

    const float* data = reinterpret_cast<const float*>(tok_a->data());
    ASSERT_TRUE(std::abs(data[0] - 1.0f) < 1e-6f);
    ASSERT_TRUE(std::abs(data[3] - 4.0f) < 1e-6f);
}

int main() {
    std::cout << "test_gguf_parser:\n";

    TEST(ggml_dtype_size);
    TEST(map_ggml_dtype);
    TEST(metadata_variant);
    TEST(get_metadata_or);
    TEST(parse_minimal_gguf);
    TEST(parse_metadata_types);
    TEST(read_tensor_data);
    TEST(invalid_magic);
    TEST(name_mapping);
    TEST(f16_to_f32);
    TEST(bf16_to_f32);
    TEST(f16_tensor_read);
    TEST(bf16_tensor_read);
    TEST(extract_config);
    TEST(shared_weight_store_tied_embeddings);

    std::cout << (tests_failed == 0 ? "All tests passed!" : "Some tests FAILED!")
              << " (" << tests_passed << " passed, " << tests_failed << " failed)\n";
    return tests_failed > 0 ? 1 : 0;
}
