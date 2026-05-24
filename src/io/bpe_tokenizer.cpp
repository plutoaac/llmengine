#include "minillm/io/bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace minillm {

// ========== GPT-2 byte <-> unicode mapping ==========

static uint32_t g_byte_to_unicode[256];
static uint8_t  g_unicode_to_byte[65536];

static void init_byte_unicode_table() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    std::vector<int> direct;
    for (int b = 33; b <= 126; ++b) direct.push_back(b);
    for (int b = 161; b <= 172; ++b) direct.push_back(b);
    for (int b = 174; b <= 255; ++b) direct.push_back(b);

    std::fill(g_unicode_to_byte, g_unicode_to_byte + 65536, uint8_t(0xFF));

    for (int b : direct) {
        g_byte_to_unicode[b] = static_cast<uint32_t>(b);
        g_unicode_to_byte[b] = static_cast<uint8_t>(b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool is_direct = false;
        for (int d : direct) { if (d == b) { is_direct = true; break; } }
        if (!is_direct) {
            uint32_t cp = 256 + n;
            g_byte_to_unicode[b] = cp;
            g_unicode_to_byte[cp] = static_cast<uint8_t>(b);
            ++n;
        }
    }
}

static std::string bytes_to_unicode(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() * 2);
    for (uint8_t b : raw) {
        uint32_t cp = g_byte_to_unicode[b];
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

static std::string unicode_to_bytes(const std::string& mapped) {
    std::string out;
    out.reserve(mapped.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.data());
    const uint8_t* end = p + mapped.size();
    while (p < end) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p; p += 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6; cp |= (p[1] & 0x3F); p += 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12; cp |= (p[1] & 0x3F) << 6; cp |= (p[2] & 0x3F); p += 3;
        } else {
            cp = (*p & 0x07) << 18; cp |= (p[1] & 0x3F) << 12; cp |= (p[2] & 0x3F) << 6; cp |= (p[3] & 0x3F); p += 4;
        }
        if (cp < 65536 && g_unicode_to_byte[cp] != 0xFF) {
            out += static_cast<char>(g_unicode_to_byte[cp]);
        }
    }
    return out;
}

// ========== UTF-8 helpers ==========

static int utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static char32_t utf8_decode(const char* s, int& len) {
    uint8_t c = static_cast<uint8_t>(*s);
    if (c < 0x80) { len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { len = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    len = 4; return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
}

static bool unicode_is_letter(char32_t cp) {
    return std::isalpha(static_cast<int>(cp));
}

static bool unicode_is_digit(char32_t cp) {
    return std::isdigit(static_cast<int>(cp));
}

static bool unicode_is_space(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}

static bool unicode_is_newline(char32_t cp) {
    return cp == '\n' || cp == '\r';
}

// ========== init ==========

std::expected<void, Status> BPETokenizer::init(const GGUFFile& file) {
    init_byte_unicode_table();

    const auto& meta = file.metadata;

    auto get_i64 = [&](const std::string& key) -> int32_t {
        auto it = meta.find(key);
        if (it == meta.end()) return -1;

        const auto& v = it->second;
        if (auto* p = std::get_if<int32_t>(&v)) return *p;
        if (auto* p = std::get_if<uint32_t>(&v)) return static_cast<int32_t>(*p);
        if (auto* p = std::get_if<int64_t>(&v)) return static_cast<int32_t>(*p);
        if (auto* p = std::get_if<uint64_t>(&v)) return static_cast<int32_t>(*p);
        if (auto* p = std::get_if<bool>(&v)) return *p ? 1 : 0;
        return -1;
    };

    bos_id_ = get_i64("tokenizer.ggml.bos_token_id");
    eos_id_ = get_i64("tokenizer.ggml.eos_token_id");
    pad_id_ = get_i64("tokenizer.ggml.padding_token_id");

    auto it = meta.find("tokenizer.ggml.tokens");
    if (it == meta.end()) {
        return std::unexpected(Status::not_found(
            "tokenizer.ggml.tokens not found in GGUF metadata"));
    }
    auto* tokens_arr = std::get_if<MetadataArray>(&it->second);
    if (!tokens_arr) {
        return std::unexpected(Status::type_error(
            "tokenizer.ggml.tokens is not an array"));
    }

    vocab_.resize(tokens_arr->elements.size());
    for (size_t i = 0; i < tokens_arr->elements.size(); ++i) {
        auto* s = std::get_if<std::string>(&tokens_arr->elements[i]);
        if (!s) {
            return std::unexpected(Status::type_error(
                "tokenizer.ggml.tokens element is not a string"));
        }
        vocab_[i] = *s;
        vocab_map_[vocab_[i]] = static_cast<int32_t>(i);
    }

    auto mit = meta.find("tokenizer.ggml.merges");
    if (mit != meta.end()) {
        auto* merges_arr = std::get_if<MetadataArray>(&mit->second);
        if (merges_arr) {
            for (size_t i = 0; i < merges_arr->elements.size(); ++i) {
                auto* s = std::get_if<std::string>(&merges_arr->elements[i]);
                if (!s) continue;
                std::string merge = *s;
                auto sp = merge.find(' ');
                if (sp == std::string::npos) continue;
                bpe_ranks_[{merge.substr(0, sp), merge.substr(sp + 1)}] = static_cast<int32_t>(i);
            }
        }
    }

    for (size_t i = 0; i < vocab_.size(); ++i) {
        if (vocab_[i].empty()) continue;
        auto pieces = pre_tokenize(vocab_[i]);
        if (pieces.size() > 1) {
            added_tokens_map_[vocab_[i]] = static_cast<int32_t>(i);
        }
    }

    added_tokens_.assign(added_tokens_map_.begin(), added_tokens_map_.end());
    std::sort(added_tokens_.begin(), added_tokens_.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    return {};
}

// ========== pre_tokenize (state machine) ==========

static int match_contraction(const char* p, const char* end) {
    if (p + 1 >= end) return 0;
    char next = p[1];
    switch (next) {
        case 's': case 't': case 'd': case 'm':
        case 'S': case 'T': case 'D': case 'M':
            return 2;
        case 'l':
            if (p + 2 < end && (p[2] == 'l' || p[2] == 'L')) return 3;
            return 0;
        case 'L':
            if (p + 2 < end && (p[2] == 'l' || p[2] == 'L')) return 3;
            return 0;
        case 'v':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        case 'V':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        case 'r':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        case 'R':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        default:
            return 0;
    }
}

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const char* p = text.data();
    const char* end = p + text.size();

    auto emit = [&](const char* from, const char* to) {
        if (from < to) pieces.emplace_back(from, to - from);
    };

    while (p < end) {
        int clen;
        char32_t cp = utf8_decode(p, clen);

        if (cp == '\'' && match_contraction(p, end) > 0) {
            int mlen = match_contraction(p, end);
            emit(p, p + mlen);
            p += mlen;
            continue;
        }

        bool is_letter = unicode_is_letter(cp);
        bool is_digit = unicode_is_digit(cp);
        bool is_space = unicode_is_space(cp);
        bool is_newline = unicode_is_newline(cp);

        if (is_letter) {
            const char* start = p;
            p += clen;
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_letter(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            emit(start, p);
            continue;
        }

        if (!is_space && !is_newline && !is_digit && !is_letter && p + clen < end) {
            int nlen;
            char32_t ncp = utf8_decode(p + clen, nlen);
            if (unicode_is_letter(ncp)) {
                const char* start = p;
                p += clen;
                while (p < end) {
                    char32_t c2 = utf8_decode(p, nlen);
                    if (unicode_is_letter(c2)) {
                        p += nlen;
                    } else {
                        break;
                    }
                }
                emit(start, p);
                continue;
            }
        }

        if (is_digit) {
            const char* start = p;
            int count = 0;
            while (p < end && count < 3) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_digit(ncp)) {
                    p += nlen;
                    count++;
                } else {
                    break;
                }
            }
            emit(start, p);
            continue;
        }

        if (!is_space && !is_newline && !is_digit && !is_letter) {
            const char* start = p;
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (!unicode_is_space(ncp) && !unicode_is_letter(ncp) && !unicode_is_digit(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            while (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                p += clen;
            }
            emit(start, p);
            continue;
        }

        if (cp == ' ') {
            if (p + 1 < end) {
                int nlen;
                char32_t ncp = utf8_decode(p + 1, nlen);
                if (!unicode_is_space(ncp) && !unicode_is_newline(ncp) &&
                    !unicode_is_letter(ncp) && !unicode_is_digit(ncp)) {
                    const char* start = p;
                    p += 1;
                    while (p < end) {
                        int nnlen;
                        char32_t nncp = utf8_decode(p, nnlen);
                        if (!unicode_is_space(nncp) && !unicode_is_letter(nncp) && !unicode_is_digit(nncp)) {
                            p += nnlen;
                        } else {
                            break;
                        }
                    }
                    while (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                        p += clen;
                    }
                    emit(start, p);
                    continue;
                }
            }
        }

        if (is_space || is_newline) {
            const char* start = p;
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_space(ncp) && !unicode_is_newline(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            if (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                while (p < end) {
                    int nlen;
                    char32_t ncp = utf8_decode(p, nlen);
                    if (unicode_is_newline(ncp)) {
                        p += nlen;
                    } else {
                        break;
                    }
                }
                emit(start, p);
                continue;
            }

            bool trailing = (p >= end) || unicode_is_space(utf8_decode(p, clen));
            p = end;
            emit(start, p);
            continue;
        }

        p += clen;
        emit(p - clen, p);
    }

    return pieces;
}

// ========== BPE ==========

struct Symbol {
    std::string text;
    int prev = -1;
    int next = -1;
};

std::vector<int32_t> BPETokenizer::bpe(const std::string& word) const {
    if (word.empty()) return {};

    std::vector<Symbol> symbols;
    const char* p = word.data();
    const char* wend = p + word.size();
    while (p < wend) {
        int len = utf8_char_len(*p);
        symbols.push_back({std::string(p, len)});
        p += len;
    }
    if (symbols.empty()) return {};

    if (symbols.size() == 1) {
        auto it = vocab_map_.find(symbols[0].text);
        return (it != vocab_map_.end()) ? std::vector<int32_t>{it->second} : std::vector<int32_t>{};
    }

    for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
        symbols[i].prev = i - 1;
        symbols[i].next = (i + 1 < static_cast<int>(symbols.size())) ? i + 1 : -1;
    }

    while (true) {
        int best_rank = INT32_MAX;
        int best_idx = -1;

        for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
            if (symbols[i].next < 0) continue;
            int j = symbols[i].next;
            auto it = bpe_ranks_.find({symbols[i].text, symbols[j].text});
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;

        int j = symbols[best_idx].next;
        symbols[best_idx].text += symbols[j].text;
        symbols[best_idx].next = symbols[j].next;
        if (symbols[j].next >= 0)
            symbols[symbols[j].next].prev = best_idx;
        symbols[j].prev = -2;
    }

    std::vector<int32_t> result;
    for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
        if (symbols[i].prev == -2) continue;
        if (symbols[i].prev != -1 && symbols[i].prev != -2) continue;
        int cur = i;
        while (cur >= 0) {
            auto it = vocab_map_.find(symbols[cur].text);
            if (it != vocab_map_.end()) result.push_back(it->second);
            cur = symbols[cur].next;
        }
        break;
    }
    return result;
}

// ========== encode ==========

std::expected<std::vector<int32_t>, Status> BPETokenizer::encode(
    const std::string& text, bool add_bos, bool add_eos) const {

    std::vector<int32_t> tokens;

    if (add_bos && bos_id_ >= 0)
        tokens.push_back(bos_id_);

    size_t pos = 0;
    while (pos < text.size()) {
        size_t earliest = std::string::npos;
        int32_t matched_id = -1;
        size_t matched_len = 0;

        for (const auto& [tok_text, tok_id] : added_tokens_) {
            size_t found = text.find(tok_text, pos);
            if (found < earliest) {
                earliest = found;
                matched_id = tok_id;
                matched_len = tok_text.size();
            }
        }

        if (earliest == std::string::npos) {
            std::string rest = text.substr(pos);
            auto pieces = pre_tokenize(rest);
            for (auto& piece : pieces) {
                std::string mapped = bytes_to_unicode(piece);
                auto ids = bpe(mapped);
                tokens.insert(tokens.end(), ids.begin(), ids.end());
            }
            break;
        }

        if (earliest > pos) {
            std::string before = text.substr(pos, earliest - pos);
            auto pieces = pre_tokenize(before);
            for (auto& piece : pieces) {
                std::string mapped = bytes_to_unicode(piece);
                auto ids = bpe(mapped);
                tokens.insert(tokens.end(), ids.begin(), ids.end());
            }
        }

        tokens.push_back(matched_id);
        pos = earliest + matched_len;
    }

    if (add_eos && eos_id_ >= 0)
        tokens.push_back(eos_id_);

    return tokens;
}

// ========== decode ==========

std::string BPETokenizer::token_to_text(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) return "";

    const std::string& tok = vocab_[id];

    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
        uint8_t byte = 0;
        for (int i = 3; i < 5; ++i) {
            byte <<= 4;
            if (tok[i] >= '0' && tok[i] <= '9') byte |= (tok[i] - '0');
            else if (tok[i] >= 'a' && tok[i] <= 'f') byte |= (tok[i] - 'a' + 10);
            else if (tok[i] >= 'A' && tok[i] <= 'F') byte |= (tok[i] - 'A' + 10);
        }
        return std::string(1, static_cast<char>(byte));
    }

    return unicode_to_bytes(tok);
}

std::expected<std::string, Status> BPETokenizer::decode(
    const std::vector<int32_t>& ids) const {

    if (vocab_.empty() && !ids.empty()) {
        return std::unexpected(Status::not_found(
            "tokenizer not initialized: vocab is empty"));
    }

    std::string result;
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) {
            return std::unexpected(Status::not_found(
                "token id out of range"));
        }
        if (id == eos_id_) break;
        result += token_to_text(id);
    }
    return result;
}

} // namespace minillm
