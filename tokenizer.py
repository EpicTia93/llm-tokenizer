"""A byte-level BPE tokenizer built from scratch.

Byte Pair Encoding (BPE) starts from the 256 possible byte values and, on each
iteration, finds the most frequent adjacent pair of tokens and merges it into a
single new token. Doing this ``vocab_size - 256`` times yields a vocabulary that
captures common substrings (word fragments, whole words, common phrases).

This implementation does GPT-style *regex pre-tokenization*: the text is first
split into chunks (words, numbers, punctuation runs, whitespace) and BPE merges
only ever happen *within* a chunk, never across chunk boundaries. That mirrors
how OpenAI's ``cl100k_base`` works and keeps merges linguistically sensible.

The design follows Andrej Karpathy's "minbpe", reimplemented here for learning.
"""

import regex as re

# GPT-4 / cl100k_base style split pattern. Requires the third-party ``regex``
# module (not the stdlib ``re``) because of the Unicode property escapes
# (\p{L}, \p{N}) and possessive quantifiers (?+, ++).
# We split each word in the pre-processing dataset according to this regex before
# tokenizing each item of the list independently
GPT_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def get_stats(ids, counts=None):
    """Count how often each adjacent pair appears in the list ``ids``.

    Pass an existing ``counts`` dict to accumulate across many chunks.
    Returns ``{(a, b): frequency}``.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Return a copy of ``ids`` with every occurrence of ``pair`` replaced by ``idx``."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class RegexBPETokenizer:
    """A regex-split, byte-level BPE tokenizer with train/encode/decode/save/load."""

    def __init__(self, pattern=GPT_SPLIT_PATTERN):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        # (int, int) -> int : the learned merges, in the order they were learned.
        self.merges = {}
        # int -> bytes : how to turn a token id back into raw bytes.
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """Reconstruct the id -> bytes table from the base bytes plus self.merges."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size, verbose=False):
        """Learn ``vocab_size - 256`` merges from ``text``."""
        assert vocab_size >= 256, "vocab_size must be at least 256 (the base bytes)"
        num_merges = vocab_size - 256

        # Split into chunks, then represent each chunk as a list of byte values.
        chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # Count pair frequencies across every chunk.
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            if not stats:
                # Nothing left to merge (text too short for the requested vocab).
                if verbose:
                    print(f"stopping early at {i} merges: no pairs left to merge")
                break

            # The most frequent pair becomes the next new token.
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                count = stats[pair]
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]!r}) had {count} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, ids):
        """Greedily apply learned merges to one chunk's byte ids."""
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Pick the pair whose merge was learned earliest (lowest id). Pairs
            # that were never a merge get +inf and are ignored.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no more applicable merges
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        """Encode a string into a list of token ids."""
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    # ----------------------------------------------------------------- decode
    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save / load
    def save(self, prefix):
        """Write ``{prefix}.model`` (machine-readable) and ``{prefix}.vocab`` (human-readable)."""
        model_path = f"{prefix}.model"
        with open(model_path, "w", encoding="utf-8") as f:
            f.write("regex-bpe v1\n")
            f.write(self.pattern + "\n")
            # One "idx1 idx2" per merge, in learned order. The line's position
            # implies the new id (256 + line number), so it need not be stored.
            for (p0, p1) in self.merges:
                f.write(f"{p0} {p1}\n")

        vocab_path = f"{prefix}.vocab"
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # errors="replace" so partial-utf8 tokens still render.
                rendered = token.decode("utf-8", errors="replace")
                f.write(f"[{rendered}] {idx}\n")
        return model_path

    @classmethod
    def load(cls, model_path):
        """Reconstruct a tokenizer from a ``.model`` file written by :meth:`save`."""
        merges = {}
        with open(model_path, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "regex-bpe v1", f"unexpected model version: {version}"
            pattern = f.readline().rstrip("\n")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p0, p1 = map(int, line.split())
                # New id is derived from how many merges precede this one, so
                # the mapping stays correct even if blank lines sneak in.
                merges[(p0, p1)] = 256 + len(merges)

        tok = cls(pattern=pattern)
        tok.merges = merges
        tok.vocab = tok._build_vocab()
        return tok
