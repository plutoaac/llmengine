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
    // Without init, encode returns empty token list (no vocabulary to look up)
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
    // Empty input should either return empty string or error
    CHECK(result.has_value(), "decode empty ids should succeed");
    CHECK(result->empty(), "decode empty ids should produce empty string");
}

void test_tokenizer_init_from_gguf_bad_file() {
    BPETokenizer tokenizer;
    GGUFFile file; // empty/default file
    auto result = tokenizer.init(file);
    // Should fail because there are no tokens in an empty GGUFFile
    CHECK(!result.has_value(), "init from empty GGUFFile should fail");
}

int main() {
    test_tokenizer_empty_init();
    test_tokenizer_encode_without_init();
    test_tokenizer_decode_without_init();
    test_tokenizer_decode_empty_ids();
    test_tokenizer_init_from_gguf_bad_file();

    std::cout << "BPETokenizer tests: " << tests_passed << "/" << tests_run << " passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
