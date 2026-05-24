#include <cstdint>
#include <expected>
#include <iostream>
#include <string>
#include <vector>

#include "minillm/minillm.h"
#include "minillm/io/gguf_weight_loader.h"
#include "minillm/io/bpe_tokenizer.h"

using namespace minillm;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [prompt]\n";
        return 1;
    }

    std::string gguf_path = argv[1];
    std::string prompt = (argc >= 3) ? argv[2] : "Hello";

    WeightLoader loader(gguf_path);
    auto file_result = loader.open();
    if (!file_result) {
        std::cerr << "Failed to parse GGUF: " << file_result.error().to_string() << "\n";
        return 1;
    }

    BPETokenizer tokenizer;
    auto tok_st = tokenizer.init(*file_result);
    if (!tok_st) {
        std::cerr << "Failed to init tokenizer: " << tok_st.error().to_string() << "\n";
        return 1;
    }

    std::cerr << "Tokenizer vocab size: " << tokenizer.vocab_size() << "\n";
    std::cerr << "BOS token id: " << tokenizer.bos_token_id() << "\n";
    std::cerr << "EOS token id: " << tokenizer.eos_token_id() << "\n";

    // Build the same prompt as generate.cpp
    std::string chat_prompt =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::cout << "=== Chat prompt ===\n";
    std::cout << chat_prompt << "\n";
    std::cout << "=== End prompt ===\n\n";

    // Encode with our tokenizer
    auto encoded = tokenizer.encode(chat_prompt, false, false);
    if (!encoded) {
        std::cerr << "Failed to encode: " << encoded.error().to_string() << "\n";
        return 1;
    }

    std::cout << "=== Tokens (" << encoded->size() << " tokens) ===\n";
    for (size_t i = 0; i < encoded->size(); ++i) {
        int32_t id = (*encoded)[i];
        std::string token_text = tokenizer.decode({id}) ? *tokenizer.decode({id}) : "<decode_error>";
        // Escape newlines and special chars for readability
        std::string escaped;
        for (char c : token_text) {
            if (c == '\n') escaped += "\\n";
            else if (c == '\t') escaped += "\\t";
            else if (c == '\r') escaped += "\\r";
            else escaped += c;
        }
        std::cout << "  [" << i << "] id=" << id << " text=\"" << escaped << "\"\n";
    }

    // Also encode without chat template for comparison
    std::cout << "\n=== Without chat template (just prompt) ===\n";
    auto plain = tokenizer.encode(prompt, false, false);
    if (plain) {
        std::cout << "  Tokens: ";
        for (size_t i = 0; i < plain->size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << (*plain)[i];
        }
        std::cout << "\n  Count: " << plain->size() << "\n";
    }

    return 0;
}