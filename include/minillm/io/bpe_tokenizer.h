#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/io/gguf_parser.h"

namespace minillm {

struct BPERank {
    int32_t id;
    float score;
};

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Build tokenizer from parsed GGUF metadata.
    std::expected<void, Status> init(const GGUFFile& file);

    // Encode text to token IDs.
    std::expected<std::vector<int32_t>, Status> encode(
        const std::string& text, bool add_bos = false, bool add_eos = false) const;

    // Decode token IDs back to text.
    std::expected<std::string, Status> decode(
        const std::vector<int32_t>& ids) const;

    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(tokens_.size()); }

private:
    // GPT-2 BPE: byte -> unicode char mapping
    static std::string bytes_to_unicode();
    static std::unordered_map<uint8_t, char32_t> byte_to_unicode_map();

    // Pre-tokenization regex pattern (GPT-2 style)
    std::vector<std::string> gpt2_pretokenize(const std::string& text) const;

    // Core BPE merge on a single word (sequence of unicode chars)
    std::vector<int32_t> bpe(const std::string& token_str) const;

    // Token table
    std::vector<std::string> tokens_;
    std::vector<int32_t> token_types_;

    // token string -> id
    std::unordered_map<std::string, int32_t> token_to_id_;

    // BPE merge ranks: "left right" -> priority (lower = higher priority)
    std::unordered_map<std::string, int32_t> merge_ranks_;

    // Byte-level encoding helpers
    std::string byte_encoder_[256];
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
};

} // namespace minillm
