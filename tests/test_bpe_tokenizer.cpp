#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "minillm/io/bpe_tokenizer.h"
#include "minillm/io/gguf_parser.h"

using namespace minillm;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) \
    do { \
        ++tests_run; \
        if (cond) { ++tests_passed; } \
        else { std::cerr << "FAIL: " << msg << " at line " << __LINE__ << "\n"; } \
    } while(0)

void test_tokenizer_empty_init() {
    BPETokenizer tokenizer;
    CHECK(tokenizer.vocab_size() == 0, "empty tokenizer vocab_size should be 0");
    CHECK(tokenizer.bos_token_id() == -1, "empty tokenizer bos should be -1");
    CHECK(tokenizer.eos_token_id() == -1, "empty tokenizer eos should be -1");
}

void test_tokenizer_encode_without_init() {
    BPETokenizer tokenizer;
    auto result = tokenizer.encode("hello");
    CHECK(result.has_value(), "encode without init should return value");
    CHECK(result->empty(), "encode without init should produce empty token list");
}

void test_tokenizer_decode_without_init() {
    BPETokenizer tokenizer;
    auto result = tokenizer.decode({1, 2, 3});
    CHECK(!result.has_value(), "decode without init should return error");
}

void test_tokenizer_decode_empty_ids() {
    BPETokenizer tokenizer;
    auto result = tokenizer.decode({});
    CHECK(result.has_value(), "decode empty ids should succeed");
    CHECK(result->empty(), "decode empty ids should produce empty string");
}

void test_tokenizer_init_from_gguf_bad_file() {
    BPETokenizer tokenizer;
    GGUFFile file;
    auto result = tokenizer.init(file);
    CHECK(!result.has_value(), "init from empty GGUFFile should fail");
}

GGUFFile make_minimal_vocab_file() {
    GGUFFile file;

    MetadataArray tokens_arr;
    tokens_arr.elements = {
        std::string("!"),
        std::string("\""),
        std::string("#"),
        std::string("$"),
        std::string("%"),
        std::string("&"),
        std::string("'"),
        std::string("("),
        std::string(")"),
        std::string("*"),
        std::string("+"),
        std::string(","),
        std::string("-"),
        std::string("."),
        std::string("/"),
        std::string("0"),
        std::string("1"),
        std::string("2"),
        std::string("3"),
        std::string("4"),
        std::string("5"),
        std::string("6"),
        std::string("7"),
        std::string("8"),
        std::string("9"),
        std::string(":"),
        std::string(";"),
        std::string("<"),
        std::string("="),
        std::string(">"),
        std::string("?"),
        std::string("@"),
        std::string("A"),
        std::string("B"),
        std::string("C"),
        std::string("D"),
        std::string("E"),
        std::string("F"),
        std::string("G"),
        std::string("H"),
        std::string("I"),
        std::string("J"),
        std::string("K"),
        std::string("L"),
        std::string("M"),
        std::string("N"),
        std::string("O"),
        std::string("P"),
        std::string("Q"),
        std::string("R"),
        std::string("S"),
        std::string("T"),
        std::string("U"),
        std::string("V"),
        std::string("W"),
        std::string("X"),
        std::string("Y"),
        std::string("Z"),
        std::string("["),
        std::string("\\"),
        std::string("]"),
        std::string("^"),
        std::string("_"),
        std::string("`"),
        std::string("a"),
        std::string("b"),
        std::string("c"),
        std::string("d"),
        std::string("e"),
        std::string("f"),
        std::string("g"),
        std::string("h"),
        std::string("i"),
        std::string("j"),
        std::string("k"),
        std::string("l"),
        std::string("m"),
        std::string("n"),
        std::string("o"),
        std::string("p"),
        std::string("q"),
        std::string("r"),
        std::string("s"),
        std::string("t"),
        std::string("u"),
        std::string("v"),
        std::string("w"),
        std::string("x"),
        std::string("y"),
        std::string("z"),
        std::string("{"),
        std::string("|"),
        std::string("}"),
        std::string("~"),
        std::string("\xc4\x80"),  // Ā  -- byte 0
        std::string("\xc4\x81"),  // ā  -- byte 1
        std::string("\xc4\x82"),  // Ă  -- byte 2
        std::string("\xc4\x83"),  // ă  -- byte 3
        std::string("\xc4\x84"),  // Ą  -- byte 4
        std::string(" "),
        std::string("<|endoftext|>"),
    };
    file.metadata["tokenizer.ggml.tokens"] = tokens_arr;

    MetadataArray merges_arr;
    merges_arr.elements = {
        std::string("t h"),
        std::string("i n"),
        std::string("t he"),
        std::string("h e"),
        std::string("h el"),
        std::string("he ll"),
        std::string("hell o"),
        std::string("e r"),
        std::string("a t"),
        std::string("o u"),
        std::string("i t"),
        std::string("v e"),
        std::string("a n"),
        std::string("o n"),
        std::string("e s"),
        std::string("i s"),
        std::string("o r"),
        std::string("e n"),
        std::string("c e"),
        std::string("e d"),
        std::string("a r"),
    };
    file.metadata["tokenizer.ggml.merges"] = merges_arr;

    file.metadata["tokenizer.ggml.eos_token_id"] = int32_t(100);
    file.metadata["tokenizer.ggml.bos_token_id"] = int32_t(99);
    file.metadata["tokenizer.ggml.padding_token_id"] = int32_t(98);

    return file;
}

void test_tokenizer_init_with_minimal_vocab() {
    BPETokenizer tokenizer;
    auto file = make_minimal_vocab_file();
    auto result = tokenizer.init(file);
    CHECK(result.has_value(), "init with minimal vocab should succeed");
    CHECK(tokenizer.vocab_size() == 101, "minimal vocab should have 101 tokens");
    CHECK(tokenizer.eos_token_id() == 100, "eos_token_id should be 100");
    CHECK(tokenizer.bos_token_id() == 99, "bos_token_id should be 99");
    CHECK(tokenizer.pad_token_id() == 98, "pad_token_id should be 98");
}

void test_tokenizer_encode_ascii() {
    BPETokenizer tokenizer;
    auto file = make_minimal_vocab_file();
    auto result = tokenizer.init(file);
    CHECK(result.has_value(), "init for encode test should succeed");

    auto encoded = tokenizer.encode("hello", false, false);
    CHECK(encoded.has_value(), "encode hello should succeed");
    CHECK(!encoded->empty(), "encode hello should produce tokens");
}

void test_tokenizer_round_trip() {
    BPETokenizer tokenizer;
    auto file = make_minimal_vocab_file();
    auto result = tokenizer.init(file);
    CHECK(result.has_value(), "init for round-trip should succeed");

    auto encoded = tokenizer.encode("Hello world", false, false);
    CHECK(encoded.has_value(), "encode should succeed");
    CHECK(!encoded->empty(), "encode should produce tokens");

    auto decoded = tokenizer.decode(*encoded);
    CHECK(decoded.has_value(), "decode should succeed");
    CHECK(!decoded->empty(), "decode should produce text");
}

void test_tokenizer_add_bos_eos() {
    BPETokenizer tokenizer;
    auto file = make_minimal_vocab_file();
    auto result = tokenizer.init(file);
    CHECK(result.has_value(), "init for bos/eos test should succeed");

    auto encoded = tokenizer.encode("A", true, true);
    CHECK(encoded.has_value(), "encode with bos/eos should succeed");
    CHECK(encoded->size() >= 3, "encode with bos+eos should have at least 3 tokens (bos + A + eos)");
    CHECK((*encoded)[0] == 99, "first token should be bos (99)");
    CHECK((*encoded).back() == 100, "last token should be eos (100)");
}

void test_tokenizer_decode_id_range() {
    BPETokenizer tokenizer;
    auto file = make_minimal_vocab_file();
    auto result = tokenizer.init(file);
    CHECK(result.has_value(), "init for range test should succeed");

    auto bad = tokenizer.decode({99999});
    CHECK(!bad.has_value(), "decode out-of-range id should fail");

    // Token 0 = '!'. EOS = id 100. Decode should stop at EOS.
    auto eos_decoded = tokenizer.decode({0, 100, 1});
    CHECK(eos_decoded.has_value(), "decode with eos should succeed");
    CHECK(*eos_decoded == "!", "decode should stop at eos token");
}

int main() {
    test_tokenizer_empty_init();
    test_tokenizer_encode_without_init();
    test_tokenizer_decode_without_init();
    test_tokenizer_decode_empty_ids();
    test_tokenizer_init_from_gguf_bad_file();
    test_tokenizer_init_with_minimal_vocab();
    test_tokenizer_encode_ascii();
    test_tokenizer_round_trip();
    test_tokenizer_add_bos_eos();
    test_tokenizer_decode_id_range();

    std::cout << "BPETokenizer tests: " << tests_passed << "/" << tests_run << " passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
