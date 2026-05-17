#include "minillm/io/bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <codecvt>
#include <locale>
#include <sstream>

namespace minillm {

// ===========================================================================
// GPT-2 byte-to-unicode mapping
// ===========================================================================

// The GPT-2 BPE uses a specific mapping of byte values to unicode characters
// to avoid control characters and whitespace issues.
static const int kByteToUnicode[] = {
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
    0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x010D, 0x010E, 0x010F,
    0x0110, 0x0111, 0x0112, 0x0113, 0x0114, 0x0115, 0x0116, 0x0117,
    0x0118, 0x0119, 0x011A, 0x011B, 0x011C, 0x011D, 0x011E, 0x011F,
    0x0120, '!',  '"',  '#',  '$',  '%',  '&',  '\'',
    '(',  ')',  '*',  '+',  ',',  '-',  '.',  '/',
    '0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',
    '8',  '9',  ':',  ';',  '<',  '=',  '>',  '?',
    '@',  'A',  'B',  'C',  'D',  'E',  'F',  'G',
    'H',  'I',  'J',  'K',  'L',  'M',  'N',  'O',
    'P',  'Q',  'R',  'S',  'T',  'U',  'V',  'W',
    'X',  'Y',  'Z',  '[',  '\\', ']',  '^',  '_',
    '`',  'a',  'b',  'c',  'd',  'e',  'f',  'g',
    'h',  'i',  'j',  'k',  'l',  'm',  'n',  'o',
    'p',  'q',  'r',  's',  't',  'u',  'v',  'w',
    'x',  'y',  'z',  '{',  '|',  '}',  '~',  0x0121,
    0x0122, 0x0123, 0x0124, 0x0125, 0x0126, 0x0127, 0x0128, 0x0129,
    0x012A, 0x012B, 0x012C, 0x012D, 0x012E, 0x012F, 0x0130, 0x0131,
    0x0132, 0x0133, 0x0134, 0x0135, 0x0136, 0x0137, 0x0138, 0x0139,
    0x013A, 0x013B, 0x013C, 0x013D, 0x013E, 0x013F, 0x0140, 0x0141,
    0x0142, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7,
    0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x0143, 0x00AE, 0x00AF,
    0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
    0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
    0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
    0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
    0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7,
    0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
    0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
    0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
    0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
    0x0144, 0x0145, 0x0146, 0x0147, 0x0148, 0x0149, 0x014A, 0x014B,
};

// Convert a unicode codepoint to UTF-8 string
static std::string codepoint_to_utf8(int cp) {
    std::string result;
    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

// ===========================================================================
// GPT-2 pre-tokenization pattern
// ===========================================================================

// Simplified GPT-2 pre-tokenization: split on whitespace and punctuation
// Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> BPETokenizer::gpt2_pretokenize(const std::string& text) const {
    std::vector<std::string> result;
    std::string current;
    enum class State { Start, Letters, Digits, Punct, Space };
    State state = State::Start;
    bool pending_space = false;

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        bool is_letter = std::isalpha(c);
        bool is_digit = std::isdigit(c);

        if (is_space) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            // Consume whitespace; attach leading space to next token
            pending_space = true;
            current = " ";
            ++i;
            // Skip remaining spaces unless there's a non-space following
            while (i < text.size()) {
                unsigned char nc = static_cast<unsigned char>(text[i]);
                if (nc != ' ' && nc != '\t' && nc != '\n' && nc != '\r') break;
                ++i;
            }
            continue;
        }

        // Check for contractions first
        if (c == '\'' && i + 1 < text.size()) {
            unsigned char nc = static_cast<unsigned char>(text[i + 1]);
            std::string contraction;
            if (nc == 's') contraction = "'s";
            else if (nc == 't') contraction = "'t";
            else if (nc == 'r' && i + 2 < text.size() && text[i + 2] == 'e') contraction = "'re";
            else if (nc == 'v' && i + 2 < text.size() && text[i + 2] == 'e') contraction = "'ve";
            else if (nc == 'm') contraction = "'m";
            else if (nc == 'l' && i + 2 < text.size() && text[i + 2] == 'l') contraction = "'ll";
            else if (nc == 'd') contraction = "'d";

            if (!contraction.empty()) {
                if (!current.empty()) {
                    result.push_back(current);
                    current.clear();
                }
                result.push_back(contraction);
                i += contraction.size();
                pending_space = false;
                continue;
            }
        }

        if (!pending_space && current.empty() && result.empty() && is_letter) {
            // Start of text, no leading space
        }

        if (is_letter) {
            if (state != State::Letters && !current.empty()) {
                result.push_back(current);
                current.clear();
            }
            state = State::Letters;
            current += static_cast<char>(c);
            pending_space = false;
        } else if (is_digit) {
            if (state != State::Digits && !current.empty()) {
                result.push_back(current);
                current.clear();
            }
            state = State::Digits;
            current += static_cast<char>(c);
            pending_space = false;
        } else {
            if (state != State::Punct && !current.empty()) {
                result.push_back(current);
                current.clear();
            }
            state = State::Punct;
            current += static_cast<char>(c);
            pending_space = false;
        }
        ++i;
    }

    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

// ===========================================================================
// BPE core
// ===========================================================================

std::vector<int32_t> BPETokenizer::bpe(const std::string& token_str) const {
    if (token_str.empty()) return {};

    // Check if the entire string is a known special/added token
    auto it = token_to_id_.find(token_str);
    if (it != token_to_id_.end()) {
        return {it->second};
    }

    // Split into individual UTF-8 characters (each is a BPE "piece")
    std::vector<std::string> pieces;
    for (size_t i = 0; i < token_str.size(); ) {
        unsigned char c = static_cast<unsigned char>(token_str[i]);
        size_t char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        char_len = std::min(char_len, token_str.size() - i);
        pieces.emplace_back(token_str, i, char_len);
        i += char_len;
    }

    if (pieces.size() <= 1) {
        std::vector<int32_t> ids;
        for (const auto& p : pieces) {
            auto pit = token_to_id_.find(p);
            if (pit != token_to_id_.end()) ids.push_back(pit->second);
        }
        return ids;
    }

    // Iteratively merge the highest-priority (lowest rank) pair
    while (pieces.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string pair = pieces[i] + " " + pieces[i + 1];
            auto mit = merge_ranks_.find(pair);
            if (mit != merge_ranks_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_idx = i;
            }
        }

        if (best_rank == INT32_MAX) break;

        // Merge the pair at best_idx
        pieces[best_idx] += pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }

    // Convert pieces to token IDs
    std::vector<int32_t> ids;
    for (const auto& p : pieces) {
        auto pit = token_to_id_.find(p);
        if (pit != token_to_id_.end()) {
            ids.push_back(pit->second);
        }
    }
    return ids;
}

// ===========================================================================
// Init from GGUF
// ===========================================================================

std::expected<void, Status> BPETokenizer::init(const GGUFFile& file) {
    // Read tokens
    auto tokens_val = get_metadata(file.metadata, "tokenizer.ggml.tokens");
    if (!tokens_val) {
        return std::unexpected(Status::not_found(
            "tokenizer.ggml.tokens not found in GGUF metadata"));
    }
    auto* tokens_arr = std::get_if<MetadataArray>(&*tokens_val);
    if (!tokens_arr) {
        return std::unexpected(Status::type_error(
            "tokenizer.ggml.tokens is not an array"));
    }

    tokens_.reserve(tokens_arr->elements.size());
    for (const auto& elem : tokens_arr->elements) {
        auto* s = std::get_if<std::string>(&elem);
        if (!s) {
            return std::unexpected(Status::type_error(
                "tokenizer.ggml.tokens element is not a string"));
        }
        tokens_.push_back(*s);
        token_to_id_[*s] = static_cast<int32_t>(tokens_.size() - 1);
    }

    // Read token types (optional)
    auto types_val = get_metadata(file.metadata, "tokenizer.ggml.token_type");
    if (types_val) {
        auto* types_arr = std::get_if<MetadataArray>(&*types_val);
        if (types_arr) {
            for (const auto& elem : types_arr->elements) {
                auto* v = std::get_if<int32_t>(&elem);
                token_types_.push_back(v ? *v : 0);
            }
        }
    }

    // Read merges
    auto merges_val = get_metadata(file.metadata, "tokenizer.ggml.merges");
    if (!merges_val) {
        return std::unexpected(Status::not_found(
            "tokenizer.ggml.merges not found in GGUF metadata"));
    }
    auto* merges_arr = std::get_if<MetadataArray>(&*merges_val);
    if (!merges_arr) {
        return std::unexpected(Status::type_error(
            "tokenizer.ggml.merges is not an array"));
    }

    for (int32_t i = 0; i < static_cast<int32_t>(merges_arr->elements.size()); ++i) {
        auto* s = std::get_if<std::string>(&merges_arr->elements[i]);
        if (!s) continue;
        merge_ranks_[*s] = i;
    }

    // Build byte encoder/decoder (GPT-2 style)
    for (int b = 0; b < 256; ++b) {
        std::string encoded = codepoint_to_utf8(kByteToUnicode[b]);
        byte_encoder_[b] = encoded;
        byte_decoder_[encoded] = static_cast<uint8_t>(b);
    }

    // Read special token IDs (try multiple integer types since values can be large)
    auto read_token_id = [&](const std::string& key) -> int32_t {
        if (auto v = get_metadata_or<int32_t>(file.metadata, key, -1); v >= 0) return v;
        if (auto v = get_metadata_or<uint32_t>(file.metadata, key, static_cast<uint32_t>(-1));
            v != static_cast<uint32_t>(-1)) return static_cast<int32_t>(v);
        if (auto v = get_metadata_or<int64_t>(file.metadata, key, -1); v >= 0) return static_cast<int32_t>(v);
        if (auto v = get_metadata_or<uint64_t>(file.metadata, key, static_cast<uint64_t>(-1));
            v != static_cast<uint64_t>(-1)) return static_cast<int32_t>(v);
        return -1;
    };
    eos_id_ = read_token_id("tokenizer.ggml.eos_token_id");
    bos_id_ = read_token_id("tokenizer.ggml.bos_token_id");
    pad_id_ = read_token_id("tokenizer.ggml.padding_token_id");

    return {};
}

// ===========================================================================
// Encode
// ===========================================================================

std::expected<std::vector<int32_t>, Status> BPETokenizer::encode(
    const std::string& text, bool add_bos, bool add_eos) const {

    std::vector<int32_t> ids;

    if (add_bos && bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }

    // First, split on special tokens (anything matching <|...|> pattern)
    std::vector<std::string> segments;
    size_t pos = 0;
    while (pos < text.size()) {
        // Look for <|...|> pattern
        auto start = text.find("<|", pos);
        if (start == std::string::npos) {
            segments.push_back(text.substr(pos));
            break;
        }
        if (start > pos) {
            segments.push_back(text.substr(pos, start - pos));
        }
        auto end = text.find("|>", start);
        if (end == std::string::npos) {
            segments.push_back(text.substr(pos));
            break;
        }
        segments.push_back(text.substr(start, end + 2 - start));
        pos = end + 2;
    }

    for (const auto& segment : segments) {
        // Check if this entire segment is a special token
        auto sit = token_to_id_.find(segment);
        if (sit != token_to_id_.end() && segment.starts_with("<|")) {
            ids.push_back(sit->second);
            continue;
        }

        // Pre-tokenize the segment
        auto words = gpt2_pretokenize(segment);

        for (const auto& word : words) {
            // Byte-encode the word
            std::string byte_encoded;
            for (unsigned char c : word) {
                byte_encoded += byte_encoder_[c];
            }

            // BPE on the byte-encoded string
            auto word_ids = bpe(byte_encoded);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }
    }

    if (add_eos && eos_id_ >= 0) {
        ids.push_back(eos_id_);
    }

    return ids;
}

// ===========================================================================
// Decode
// ===========================================================================

std::expected<std::string, Status> BPETokenizer::decode(
    const std::vector<int32_t>& ids) const {

    std::string byte_encoded;
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(tokens_.size())) {
            return std::unexpected(Status::not_found(
                "token id out of range: " + std::to_string(id)));
        }
        byte_encoded += tokens_[id];
    }

    // Decode byte-encoded string back to UTF-8
    std::string result;
    std::string acc;
    for (size_t i = 0; i < byte_encoded.size(); ) {
        // Try to decode multi-byte UTF-8 sequences
        acc.clear();
        // Accumulate until we have a valid byte_decoder key
        for (size_t j = i; j < byte_encoded.size(); ++j) {
            acc += byte_encoded[j];
            auto it = byte_decoder_.find(acc);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
                i = j + 1;
                goto next_char;
            }
        }
        // Fallback: output raw byte
        result += byte_encoded[i];
        ++i;
        next_char:;
    }

    return result;
}

} // namespace minillm
