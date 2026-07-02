"""Compare our BPE tokenizer against OpenAI's ``cl100k_base`` on the same text.

The headline metric is how many tokens each tokenizer produces for a piece of
text (fewer tokens = better compression). We report bytes-per-token and
chars-per-token so texts of different lengths stay comparable.

By default we evaluate on two slices of the training corpus:
  * "held-out tail (unseen)" — the last 10%, which ``train.py`` did NOT train on
  * "training split (seen)"  — the first 90%, which it DID train on
Comparing the two exposes how much our small tokenizer overfits its corpus.

Usage:
    uv run python compare.py                      # seen vs unseen slices of the corpus
    uv run python compare.py --text "hello!"      # compare on a literal string
    uv run python compare.py --input some.txt     # compare on your own file
"""

import argparse
import os

import tiktoken

from tokenizer import RegexBPETokenizer

# Must match train.py so the "unseen" tail really was excluded from training.
HOLDOUT_FRAC = 0.1
DEFAULT_MODEL = "models/tok1024.model"
DEFAULT_CORPUS = "data/input.txt"


def measure(name, vocab_size, encode, decode, text):
    """Encode ``text`` and gather token count + compression stats."""
    ids = encode(text)
    n_tokens = len(ids)
    return {
        "name": name,
        "vocab_size": vocab_size,
        "tokens": n_tokens,
        "bytes_per_token": len(text.encode("utf-8")) / n_tokens if n_tokens else 0.0,
        "chars_per_token": len(text) / n_tokens if n_tokens else 0.0,
        "roundtrip": decode(ids) == text,
    }


def print_report(label, text, rows):
    n_chars, n_bytes = len(text), len(text.encode("utf-8"))
    print(f"\nText: {label} — {n_chars:,} chars / {n_bytes:,} bytes")
    header = (
        f"{'Tokenizer':<16}{'Vocab size':>12}{'Tokens':>12}"
        f"{'Bytes/token':>14}{'Chars/token':>14}{'Round-trip':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<16}{r['vocab_size']:>12,}{r['tokens']:>12,}"
            f"{r['bytes_per_token']:>14.2f}{r['chars_per_token']:>14.2f}"
            f"{('OK' if r['roundtrip'] else 'FAIL'):>12}"
        )
    ours, ref = rows[0], rows[1]
    if ref["tokens"]:
        diff = (ours["tokens"] - ref["tokens"]) / ref["tokens"] * 100
        word = "more" if diff > 0 else "fewer"
        print(
            f"=> ours uses {abs(diff):.1f}% {word} tokens than cl100k_base "
            f"(vocab {ours['vocab_size']:,} vs {ref['vocab_size']:,})."
        )


def build_scenarios(args):
    """Return a list of (label, text) pairs to evaluate."""
    if args.text is not None:
        return [("custom text", args.text)]
    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as f:
            return [(args.input, f.read())]
    # Default: seen vs unseen slices of the training corpus.
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * (1 - HOLDOUT_FRAC))
    return [
        ("held-out tail (unseen)", text[split:]),
        ("training split (seen)", text[:split]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Compare our BPE tokenizer vs cl100k_base.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to our .model file")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus for the seen/unseen split")
    parser.add_argument("--input", default=None, help="evaluate on this file instead")
    parser.add_argument("--text", default=None, help="evaluate on this literal string instead")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}\nTrain one first: uv run python train.py")

    ours = RegexBPETokenizer.load(args.model)
    ref = tiktoken.get_encoding("cl100k_base")

    for label, text in build_scenarios(args):
        rows = [
            measure("ours (regex)", len(ours.vocab), ours.encode, ours.decode, text),
            measure("cl100k_base", ref.n_vocab, ref.encode, ref.decode, text),
        ]
        print_report(label, text, rows)

    print(
        "\nNote: cl100k_base compresses better because it has a far larger "
        "vocabulary and\nwas trained on vastly more text. Our tokenizer looks "
        "best on its own training\nsplit (in-sample) and worse on the held-out "
        "tail — that gap is overfitting."
    )


if __name__ == "__main__":
    main()
