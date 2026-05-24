#pragma once

#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "minillm/core/status.h"
#include "minillm/io/gguf_parser.h"

namespace minillm {

class BPETokenizer {
public:
    BPETokenizer() = default;

    std::expected<void, Status> init(const GGUFFile& file);

    std::expected<std::vector<int32_t>, Status> encode(
        const std::string& text, bool add_bos = false, bool add_eos = false) const;

    std::expected<std::string, Status> decode(
        const std::vector<int32_t>& ids) const;

    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(vocab_.size()); }

private:
    std::string token_to_text(int32_t id) const;
    std::vector<int32_t> bpe(const std::string& word) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> vocab_map_;
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;

    int32_t eos_id_ = -1;
    int32_t bos_id_ = -1;
    int32_t pad_id_ = -1;

    std::unordered_map<std::string, int32_t> added_tokens_map_;
    std::vector<std::pair<std::string, int32_t>> added_tokens_;
};

} // namespace minillm
