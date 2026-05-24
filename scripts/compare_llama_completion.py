#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROMPT_TOKEN_RE = re.compile(r"Encoded to (\d+) tokens")
LLAMA_PROMPT_TOKEN_RE = re.compile(r"prompt eval time = .* / (\d+) tokens")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


def extract_visible_text(text: str, stop_prefixes: tuple[str, ...]) -> str:
    lines: list[str] = []
    for raw_line in strip_ansi(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in stop_prefixes):
            break
        if line.startswith("[Generated"):
            break
        lines.append(line)
    return "\n".join(lines).replace("[end of text]", "").strip()


def run_plain(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def run_with_pty(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, float]:
    import pty

    start = time.perf_counter()
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

        chunks: list[bytes] = []
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)

        rc = proc.wait()
        elapsed = time.perf_counter() - start
        output = b"".join(chunks).decode("utf-8", "replace")
        return rc, output, elapsed
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MiniLLMEngine and llama.cpp on one prompt.")
    parser.add_argument("model", help="Path to the GGUF model")
    parser.add_argument("prompt", nargs="?", default="Hello", help="User prompt to compare")
    parser.add_argument("max_tokens", nargs="?", type=int, default=32, help="Number of generated tokens")
    parser.add_argument(
        "--mini-bin",
        default=None,
        help="Path to MiniLLMEngine's generate binary (default: repo/build/generate)",
    )
    parser.add_argument(
        "--llama-bin",
        default=None,
        help="Path to llama.cpp's llama-completion binary",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt used for both sides",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mini_bin = Path(args.mini_bin) if args.mini_bin else repo_root / "build" / "generate"
    llama_bin = Path(args.llama_bin) if args.llama_bin else repo_root.parent / "llama.cpp-master" / "build" / "bin" / "llama-completion"

    if not mini_bin.exists():
        print(f"MiniLLMEngine binary not found: {mini_bin}", file=sys.stderr)
        return 1
    if not llama_bin.exists():
        print(f"llama.cpp binary not found: {llama_bin}", file=sys.stderr)
        return 1

    raw_prompt = (
        f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False) as fp:
        fp.write(raw_prompt)
        prompt_file = Path(fp.name)

    try:
        mini_rc, mini_stdout, mini_stderr, mini_elapsed = run_plain(
            [str(mini_bin), args.model, args.prompt, str(args.max_tokens)],
            cwd=repo_root,
        )
        if mini_rc != 0:
            print(mini_stdout, end="")
            print(mini_stderr, end="", file=sys.stderr)
            return mini_rc

        llama_rc, llama_output, llama_elapsed = run_with_pty(
            [
                str(llama_bin),
                "-m",
                args.model,
                "-f",
                str(prompt_file),
                "-n",
                str(args.max_tokens),
                "-s",
                "42",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1",
                "--repeat-penalty",
                "1.1",
                "--simple-io",
                "--no-display-prompt",
                "--skip-chat-parsing",
                "--log-disable",
                "-no-cnv",
            ],
            cwd=llama_bin.parent,
        )
        if llama_rc != 0:
            print(llama_output, end="", file=sys.stderr)
            return llama_rc

        mini_text = extract_visible_text(mini_stdout, ("[Generated",))
        llama_text = extract_visible_text(llama_output, ("common_perf_print:", "common_memory_breakdown_print:"))

        mini_prompt_tokens = None
        llama_prompt_tokens = None

        mini_prompt_match = PROMPT_TOKEN_RE.search(strip_ansi(mini_stderr))
        if mini_prompt_match:
            mini_prompt_tokens = int(mini_prompt_match.group(1))

        llama_prompt_match = LLAMA_PROMPT_TOKEN_RE.search(strip_ansi(llama_output))
        if llama_prompt_match:
            llama_prompt_tokens = int(llama_prompt_match.group(1))

        print(f"MiniLLMEngine elapsed: {mini_elapsed * 1000.0:.2f} ms")
        print(f"llama.cpp elapsed: {llama_elapsed * 1000.0:.2f} ms")
        if mini_prompt_tokens is not None:
            print(f"MiniLLMEngine prompt tokens: {mini_prompt_tokens}")
        if llama_prompt_tokens is not None:
            print(f"llama.cpp prompt tokens: {llama_prompt_tokens}")
        print()
        print(f"MiniLLMEngine: {mini_text}")
        print(f"llama.cpp:     {llama_text}")

        if mini_text != llama_text:
            print("\n--- diff ---")
            diff = difflib.unified_diff(
                [mini_text + "\n"],
                [llama_text + "\n"],
                fromfile="MiniLLMEngine",
                tofile="llama.cpp",
            )
            print("".join(diff), end="")
            return 1

        return 0
    finally:
        try:
            prompt_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
