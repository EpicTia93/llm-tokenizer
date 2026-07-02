# llm-tokenizer

A byte-level **Byte Pair Encoding (BPE)** tokenizer built from scratch in Python,
plus a script that compares its compression against OpenAI's `cl100k_base`.

## How BPE works here

1. **Start from raw bytes.** Any text is UTF-8 encoded, so the base vocabulary is
   the 256 possible byte values (ids `0..255`).
2. **Pre-tokenize with a regex.** Before merging, text is split into chunks
   (words, numbers, punctuation runs, whitespace) using a GPT-4/`cl100k_base`-style
   pattern. Merges only ever happen *within* a chunk, so a token never spans a
   word/space boundary — the same trick real tokenizers use.
3. **Merge the most frequent pair.** Repeatedly: count every adjacent token pair,
   take the most frequent one, mint a new token id (`256`, `257`, …) for it, and
   replace all occurrences. Do this `vocab_size - 256` times.
4. **Encode / decode.** `encode` re-applies the learned merges (earliest-learned
   first) to new text; `decode` looks each id up in the id→bytes table and UTF-8
   decodes the result.

Implementation lives in `tokenizer.py` (`RegexBPETokenizer`), following the design
of Andrej Karpathy's "minbpe", reimplemented for learning.

## Setup

Uses [uv](https://docs.astral.sh/uv/). Dependencies (`tiktoken`, `regex`) are
already declared in `pyproject.toml`:

```bash
uv sync        # create the venv and install deps (first time)
```

## Usage

```bash
# 1. Train. Downloads ~1MB of Shakespeare to data/input.txt on first run,
#    trains on the first 90% (last 10% held out), saves models/tok1024.*
uv run python train.py --vocab-size 1024 --verbose

# 2. Compare against cl100k_base on seen vs. unseen text.
uv run python compare.py

# Compare on your own text instead:
uv run python compare.py --text "Attention is all you need."
uv run python compare.py --input path/to/file.txt

# 3. Run the tests.
uv run python test_tokenizer.py
```

Train on your own corpus (any UTF-8 `.txt`, e.g. a Project Gutenberg book):

```bash
uv run python train.py --input mybook.txt --vocab-size 4096 --out models/tok4096
uv run python compare.py --model models/tok4096.model
```

## What the comparison shows

`compare.py` reports, for each piece of text, how many **tokens** each tokenizer
produces (fewer = better compression), plus bytes/token and chars/token.

- **`cl100k_base` compresses better.** It has ~100k tokens vs. our ~1k, and was
  trained on far more text, so it packs more characters into each token.
- **Our tokenizer overfits its corpus.** It looks best on the *training split*
  (text it learned from) and noticeably worse on the *held-out tail* — that gap
  is the point of reporting both.

Bigger `--vocab-size` narrows the gap on in-domain text at the cost of longer
(pure-Python) training time.

## Files

| File | Purpose |
| --- | --- |
| `tokenizer.py` | `RegexBPETokenizer`: train / encode / decode / save / load |
| `train.py` | download corpus, train, save `models/tok*.model` + `.vocab` |
| `compare.py` | compression comparison vs. `cl100k_base` |
| `test_tokenizer.py` | round-trip, tiny-train, and save/load tests |

## Notes & possible extensions

- This is the **naive** reference implementation (recompute pair counts every
  merge). Fine for ~1MB / vocab 1024; for larger runs, maintain counts
  incrementally.
- Not implemented (easy follow-ups): special tokens (e.g. `<|endoftext|>`),
  much larger vocabularies, byte-fallback edge cases.
