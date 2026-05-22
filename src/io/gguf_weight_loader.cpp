#include "minillm/io/gguf_weight_loader.h"

#include <bit>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <limits>

#if defined(MINILLM_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace minillm {

namespace {

#if defined(MINILLM_ENABLE_CUDA)
Status cuda_status(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return Status::make_ok();
    return Status::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
#endif

std::unordered_map<std::string, std::string> build_value_to_gguf_map(
    const GGUFFile& file) {
    std::unordered_map<std::string, std::string> value_to_gguf;
    for (const auto& ti : file.tensor_infos) {
        auto mapped = WeightLoader::gguf_name_to_value_name(ti.name);
        if (mapped) {
            value_to_gguf[*mapped] = ti.name;
        }
    }

    // Handle tied embeddings: if lm_head.weight is not found, map to token_embd.weight.
    if (value_to_gguf.find("lm_head.weight") == value_to_gguf.end() &&
        value_to_gguf.find("tok_embeddings.weight") != value_to_gguf.end()) {
        value_to_gguf["lm_head.weight"] = value_to_gguf["tok_embeddings.weight"];
    }
    return value_to_gguf;
}

std::unordered_map<std::string, const GGUFTensorInfo*> build_gguf_tensor_map(
    const GGUFFile& file) {
    std::unordered_map<std::string, const GGUFTensorInfo*> gguf_tensor_map;
    for (const auto& ti : file.tensor_infos) {
        gguf_tensor_map[ti.name] = &ti;
    }
    return gguf_tensor_map;
}

Status load_tensor_from_gguf(const std::string& gguf_path, const GGUFFile& file,
                             const GGUFTensorInfo& ti, Tensor& tensor) {
    if (!tensor.is_allocated()) {
        return Status::runtime_error("tensor not allocated: " + tensor.name());
    }

    auto raw_bytes_exp = ti.bytes();
    if (!raw_bytes_exp) return raw_bytes_exp.error();
    const size_t raw_bytes = *raw_bytes_exp;

    std::vector<std::byte> raw_buf(raw_bytes);
    auto st = GGUFParser::read_tensor_data(
        gguf_path, file.data_offset, ti.offset, raw_buf.data(), raw_bytes);
    if (!st.ok()) return st;

    auto num_el_exp = ti.num_elements();
    if (!num_el_exp) return num_el_exp.error();
    const size_t num_el = *num_el_exp;

    if (tensor.device().type == DeviceType::CUDA) {
#if defined(MINILLM_ENABLE_CUDA)
        std::vector<float> staging(num_el);
        auto st2 = WeightLoader::dequantize_to_f32(
            ti.dtype, raw_buf.data(), staging.data(), num_el);
        if (!st2.ok()) return st2;
        auto err = cudaMemcpy(tensor.data(), staging.data(),
                              num_el * sizeof(float), cudaMemcpyHostToDevice);
        return cuda_status(err, "copy GGUF weight to CUDA tensor failed");
#else
        return Status::unsupported(
            "cannot load GGUF weights into CUDA tensor without CUDA support");
#endif
    }

    float* dst = reinterpret_cast<float*>(tensor.data());
    return WeightLoader::dequantize_to_f32(ti.dtype, raw_buf.data(), dst, num_el);
}

Status check_compatible_weight(const Tensor& tensor, const Value& value) {
    if (tensor.shape() != value.shape) {
        return Status::shape_mismatch(
            "shared weight shape mismatch for " + value.name +
            ": tensor " + tensor.shape().to_string() +
            " vs value " + value.shape.to_string());
    }
    if (tensor.dtype() != value.dtype) {
        return Status::type_error("shared weight dtype mismatch for " + value.name);
    }
    if (tensor.device().type != value.device.type ||
        tensor.device().index != value.device.index) {
        return Status::runtime_error(
            "shared weight device mismatch for " + value.name +
            ": tensor " + tensor.device().to_string() +
            " vs value " + value.device.to_string());
    }
    return Status::make_ok();
}

} // namespace

// --- SharedWeightStore ---

Tensor* SharedWeightStore::get(const std::string& value_name) const {
    auto it = aliases_by_value_name_.find(value_name);
    return it == aliases_by_value_name_.end() ? nullptr : it->second;
}

Status SharedWeightStore::bind(const Graph& graph, RuntimeContext& ctx) const {
    for (const auto& v : graph.values()) {
        if (v.kind != ValueKind::Constant) continue;
        Tensor* tensor = get(v.name);
        if (!tensor) continue;
        auto st = check_compatible_weight(*tensor, v);
        if (!st.ok()) return st;
        st = ctx.bind(v.id, tensor);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

// --- F16/BF16 to F32 conversion ---

static float f16_to_f32(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal: normalize
        int shift = std::countl_zero(mant) - 21;
        mant <<= shift;
        exp = 1 - shift;
        uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | ((mant & 0x3FF) << 13);
        return std::bit_cast<float>(f32);
    }
    if (exp == 31) {
        uint32_t f32 = (sign << 31) | 0x7F800000 | (mant ? 0x400000 : 0);
        return std::bit_cast<float>(f32);
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return std::bit_cast<float>(f32);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    return std::bit_cast<float>(f32_bits);
}

// --- WeightLoader ---

WeightLoader::WeightLoader(std::string gguf_path)
    : gguf_path_(std::move(gguf_path)) {}

std::expected<GGUFFile, Status> WeightLoader::open() {
    return GGUFParser::parse(gguf_path_);
}

std::expected<TransformerConfig, Status> WeightLoader::extract_config(
    const GGUFFile& file) const {

    TransformerConfig cfg;

    auto arch = get_metadata_or<std::string>(
        file.metadata, "general.architecture", "llama");
    std::string prefix = arch + ".";

    // Helper: read integer from {arch}.key, fall back to llama.key
    auto read_int = [&](const std::string& key, int64_t default_val) -> int64_t {
        for (const auto& p : {prefix + key, std::string("llama.") + key}) {
            auto it = file.metadata.find(p);
            if (it == file.metadata.end()) continue;
            if (auto* v = std::get_if<int32_t>(&it->second)) return *v;
            if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
            if (auto* v = std::get_if<int64_t>(&it->second)) return *v;
            if (auto* v = std::get_if<uint64_t>(&it->second))
                return static_cast<int64_t>(*v);
        }
        return default_val;
    };

    cfg.hidden_size       = read_int("embedding_length", 768);
    cfg.intermediate_size = read_int("feed_forward_length", 2048);
    cfg.num_layers        = read_int("block_count", 1);
    cfg.num_heads         = read_int("attention.head_count", 12);
    cfg.num_kv_heads      = read_int("attention.head_count_kv", 12);
    cfg.head_dim          = read_int("attention.key_length", 64);

    // RMS norm epsilon
    {
        float eps = 1e-6f;
        for (const auto& p :
             {prefix + "attention.layer_norm_rms_epsilon",
              std::string("llama.attention.layer_norm_rms_epsilon")}) {
            auto it = file.metadata.find(p);
            if (it != file.metadata.end()) {
                if (auto* v = std::get_if<float>(&it->second)) { eps = *v; break; }
                if (auto* v = std::get_if<double>(&it->second)) { eps = static_cast<float>(*v); break; }
            }
        }
        cfg.rms_norm_eps = eps;
    }

    // vocab_size: prefer tokenizer.ggml.tokens array length
    auto tok_size = get_metadata_array_size(file.metadata, "tokenizer.ggml.tokens");
    if (tok_size) {
        cfg.vocab_size = static_cast<int64_t>(*tok_size);
    } else {
        // Fall back: infer from token_embd.weight shape
        for (const auto& ti : file.tensor_infos) {
            if (ti.name == "token_embd.weight" && ti.dimensions.size() >= 1) {
                cfg.vocab_size = ti.dimensions[0];
                break;
            }
        }
    }

    cfg.batch_size = 1;
    cfg.seq_len = -1;
    cfg.dtype = DType::Float32;

    // Detect QK-norm: check if blk.0.attn_q_norm.weight exists
    for (const auto& ti : file.tensor_infos) {
        if (ti.name == "blk.0.attn_q_norm.weight") {
            cfg.use_qk_norm = true;
            break;
        }
    }

    // RoPE base frequency
    cfg.rope_base = read_int("attention.rope.freq_base", 10000);

    return cfg;
}

// --- Name mapping ---

std::expected<std::string, Status> WeightLoader::gguf_name_to_value_name(
    const std::string& gguf_name, int layer_index) {

    static const std::vector<std::pair<std::string, std::string>> layer_suffixes = {
        {"attn_norm.weight",        "attn_norm.weight"},
        {"attn_q.weight",           "q_proj.weight"},
        {"attn_k.weight",           "k_proj.weight"},
        {"attn_v.weight",           "v_proj.weight"},
        {"attn_q_norm.weight",      "q_norm.weight"},
        {"attn_k_norm.weight",      "k_norm.weight"},
        {"attn_output.weight",      "o_proj.weight"},
        {"ffn_norm.weight",         "ffn_norm.weight"},
        {"ffn_gate.weight",         "gate_proj.weight"},
        {"ffn_up.weight",           "up_proj.weight"},
        {"ffn_down.weight",         "down_proj.weight"},
    };

    static const std::vector<std::pair<std::string, std::string>> global_suffixes = {
        {"token_embd.weight",       "tok_embeddings.weight"},
        {"output_norm.weight",      "output_norm.weight"},
        {"output.weight",           "lm_head.weight"},
    };

    // Check for layer-prefixed name: "blk.{N}.suffix"
    if (gguf_name.starts_with("blk.")) {
        auto dot_pos = gguf_name.find('.', 4);
        if (dot_pos != std::string::npos) {
            std::string idx_str = gguf_name.substr(4, dot_pos - 4);
            int layer_idx = 0;
            try {
                size_t pos = 0;
                layer_idx = std::stoi(idx_str, &pos);
                if (pos != idx_str.size()) {
                    return std::unexpected(Status::invalid_argument(
                        "non-numeric layer index in GGUF tensor: " + gguf_name));
                }
            } catch (const std::exception&) {
                return std::unexpected(Status::invalid_argument(
                    "invalid layer index in GGUF tensor: " + gguf_name));
            }
            std::string suffix = gguf_name.substr(dot_pos + 1);
            for (const auto& [gguf_suffix, value_name] : layer_suffixes) {
                if (suffix == gguf_suffix) {
                    return "layer_" + std::to_string(layer_idx) + "." + std::string(value_name);
                }
            }
        }
    }

    // Check for global (non-layer) names
    for (const auto& [gguf_suffix, value_name] : global_suffixes) {
        if (gguf_name == gguf_suffix) return value_name;
    }

    return std::unexpected(Status::not_found(
        "no mapping for GGUF tensor: " + gguf_name));
}

// --- Dequantization ---

Status WeightLoader::dequantize_to_f32(
    GgmlDataType src_dtype,
    const void* src,
    float* dst,
    size_t num_elements) {

    switch (src_dtype) {
    case GgmlDataType::F32:
        std::memcpy(dst, src, num_elements * sizeof(float));
        return Status::make_ok();

    case GgmlDataType::F16: {
        const auto* src_u16 = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = f16_to_f32(src_u16[i]);
        }
        return Status::make_ok();
    }

    case GgmlDataType::BF16: {
        const auto* src_u16 = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = bf16_to_f32(src_u16[i]);
        }
        return Status::make_ok();
    }

    default:
        return Status::unsupported(
            "cannot dequantize GGML dtype " +
            std::string(ggml_dtype_name(src_dtype)));
    }
}

// --- Load weights ---

Status WeightLoader::load_weights(
    const GGUFFile& file,
    const Graph& graph,
    RuntimeContext& ctx) const {

    auto gguf_tensor_map = build_gguf_tensor_map(file);
    auto value_to_gguf = build_value_to_gguf_map(file);

    for (const auto& v : graph.values()) {
        if (v.kind != ValueKind::Constant) continue;

        auto gguf_it = value_to_gguf.find(v.name);
        if (gguf_it == value_to_gguf.end()) continue;

        const auto& ti = *gguf_tensor_map.at(gguf_it->second);
        auto* tensor = ctx.get(v.id);
        if (!tensor) {
            return Status::runtime_error(
                "tensor not found in RuntimeContext for value: " + v.name);
        }
        auto st = load_tensor_from_gguf(gguf_path_, file, ti, *tensor);
        if (!st.ok()) return st;
    }

    return Status::make_ok();
}

std::expected<SharedWeightStore, Status> WeightLoader::load_shared_weights(
    const GGUFFile& file,
    const Graph& graph) const {

    auto gguf_tensor_map = build_gguf_tensor_map(file);
    auto value_to_gguf = build_value_to_gguf_map(file);
    SharedWeightStore store;

    for (const auto& v : graph.values()) {
        if (v.kind != ValueKind::Constant) continue;

        auto gguf_it = value_to_gguf.find(v.name);
        if (gguf_it == value_to_gguf.end()) continue;

        const std::string& gguf_name = gguf_it->second;
        auto info_it = gguf_tensor_map.find(gguf_name);
        if (info_it == gguf_tensor_map.end()) {
            return std::unexpected(Status::not_found(
                "GGUF tensor not found for value: " + v.name));
        }

        Tensor* shared_tensor = nullptr;
        auto existing = store.storage_by_gguf_name_.find(gguf_name);
        if (existing == store.storage_by_gguf_name_.end()) {
            auto tensor = std::make_unique<Tensor>(v.name, v.shape, v.dtype, v.device);
            Status st;
            if (v.device.type == DeviceType::CUDA) {
                st = tensor->allocate_cuda();
            } else {
                st = tensor->allocate_cpu();
            }
            if (!st.ok()) return std::unexpected(st);

            auto bytes = tensor->nbytes();
            if (!bytes) return std::unexpected(bytes.error());

            st = load_tensor_from_gguf(gguf_path_, file, *info_it->second, *tensor);
            if (!st.ok()) return std::unexpected(st);

            shared_tensor = tensor.get();
            store.total_bytes_ += *bytes;
            store.storage_by_gguf_name_[gguf_name] = std::move(tensor);
        } else {
            shared_tensor = existing->second.get();
            auto st = check_compatible_weight(*shared_tensor, v);
            if (!st.ok()) return std::unexpected(st);
        }

        store.aliases_by_value_name_[v.name] = shared_tensor;
    }

    return std::move(store);
}

} // namespace minillm
