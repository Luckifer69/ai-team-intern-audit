"""Shared helpers for Part A: tokenizer loading and denominator (unit) counting."""

import sys
import regex

# Ensure UTF-8 output on consoles where default encoding is non-UTF-8 (e.g. Windows cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Pinned tokenizer specs used in the analysis. "hf:" specs take an optional "@revision".
TOKENIZERS = {
    "gpt2": "tiktoken:gpt2",
    "o200k": "tiktoken:o200k_base",
    "sarvam1": "hf:sarvamai/sarvam-1@e9607337286ddf496d4a2562b194e489dcf3feea",
}


def load_encoder(spec):
    """Return encode(text) -> list[int] with NO special tokens (BOS/EOS) added.

    spec: "tiktoken:<encoding>" or "hf:<repo_id>[@revision]" (tokenizer.json via `tokenizers`).
    A bare name like "gpt2" is treated as tiktoken, matching fertility.py.
    """
    spec = TOKENIZERS.get(spec, spec)
    if spec.startswith("hf:"):
        from tokenizers import Tokenizer

        repo, _, rev = spec[3:].partition("@")
        tok = Tokenizer.from_pretrained(repo, revision=rev or "main")
        return lambda s: tok.encode(s, add_special_tokens=False).ids
    import tiktoken

    enc = tiktoken.get_encoding(spec.removeprefix("tiktoken:"))
    # encode_ordinary: plain text, never interprets "<|endoftext|>" as a special token.
    return enc.encode_ordinary


def read_corpus(path):
    """Lines of a one-sentence-per-line file, split on '\\n' only, outer whitespace stripped.

    Unlike fertility.read_lines this keeps blank lines (so line i stays parallel across
    languages) and does not NFC-normalise.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [l.strip() for l in lines]


def units(text):
    """Candidate denominators for one sentence."""
    return {
        "words": len(text.split()),                      # whitespace-delimited, any Unicode space
        "graphemes": len(regex.findall(r"\X", text)),    # extended grapheme clusters (UAX #29)
        "bytes": len(text.encode("utf-8")),
        "code_points": len(text),
    }
