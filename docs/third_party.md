# Third-Party Options

MiniLLMEngine keeps the tokenizer boundary replaceable. The current default
backend is the in-tree GGUF BPE tokenizer, but these open-source options are the
most relevant if you want to swap in a library-backed implementation:

## Recommended

- `mlc-ai/tokenizers-cpp`
  - Good fit when you want a C++ wrapper around Hugging Face tokenizers or
    SentencePiece with a smaller integration surface.
  - Best match for models that ship a `tokenizer.json` / HF-style tokenizer
    config.

## Model-family specific

- `sentencepiece`
  - Mature C++ tokenizer runtime for SentencePiece-based models.
  - Best when the model family ships a `.model` file and you want native C++
    integration.

## Reference behavior

- `ggml-org/llama.cpp`
  - Not a drop-in tokenizer library for this project, but it is the most useful
    behavior reference for GGUF chat templates and model-specific tokenization.

## Why not hard-depend on one library yet?

The current project goal is an internship-ready CPU/CUDA inference engine, not a
full tokenizer zoo. Keeping a small interface lets the code stay readable while
still making future backend swaps cheap.
