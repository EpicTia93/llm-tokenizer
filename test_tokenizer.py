"""Tests for the BPE tokenizer.

Run standalone:   uv run python test_tokenizer.py
Or with pytest:   uv run python -m pytest    (uv add --dev pytest)
"""

import os
import tempfile

from tokenizer import RegexBPETokenizer, get_stats, merge

# Strings that exercise ASCII, accents, emoji, whitespace, and the empty string.
SAMPLES = [
    "",
    "hello world",
    "héllo wörld — grüße!",
    "emoji: 🙂🚀🧠 and math ∑∫",
    "tabs\tand\nnewlines\n\n  and   spaces",
    "The quick brown fox jumps over the lazy dog. 12345 67890!!!",
    "aaaaaaaaaabbbbbbbbbb",
]


def test_merge_helper():
    assert merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
    # No occurrences -> unchanged.
    assert merge([1, 2, 3], (7, 8), 99) == [1, 2, 3]
    # Overlapping pattern: only non-overlapping left-to-right matches merge.
    assert merge([6, 6, 6], (6, 6), 99) == [99, 6]


def test_get_stats():
    stats = get_stats([1, 2, 1, 2, 3])
    assert stats[(1, 2)] == 2
    assert stats[(2, 1)] == 1
    assert stats[(2, 3)] == 1


def test_roundtrip_untrained():
    # With no merges the tokenizer is just the identity over UTF-8 bytes.
    tok = RegexBPETokenizer()
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_trained():
    tok = RegexBPETokenizer()
    tok.train("the quick brown fox " * 50, vocab_size=356)
    # Round-trips even on text it was never trained on (byte fallback covers gaps).
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s


def test_tiny_train_vocab_size():
    tok = RegexBPETokenizer()
    tok.train("the quick brown fox " * 20, vocab_size=256 + 5)
    assert len(tok.merges) == 5
    assert len(tok.vocab) == 256 + 5


def test_training_compresses():
    text = "the quick brown fox " * 50
    tok = RegexBPETokenizer()
    tok.train(text, vocab_size=356)
    n_tokens = len(tok.encode(text))
    n_bytes = len(text.encode("utf-8"))
    # After learning merges, the training text needs fewer tokens than raw bytes.
    assert n_tokens < n_bytes


def test_save_load_roundtrip():
    tok = RegexBPETokenizer()
    tok.train("the quick brown fox " * 50, vocab_size=356)
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "tok")
        tok.save(prefix)
        loaded = RegexBPETokenizer.load(prefix + ".model")
    assert loaded.merges == tok.merges
    for s in SAMPLES:
        assert loaded.encode(s) == tok.encode(s)
        assert loaded.decode(loaded.encode(s)) == s


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
