"""Train the BPE tokenizer on a text corpus and save the learned model.

Usage:
    uv run python train.py                       # download Tiny Shakespeare, train to vocab 1024
    uv run python train.py --vocab-size 4096     # more merges -> better compression
    uv run python train.py --input mybook.txt    # train on your own text
"""

import argparse
import os
import time
import urllib.request

from tokenizer import RegexBPETokenizer

# ~1.1 MB of English (Shakespeare's plays). Any plain-text file works too, e.g.
# a Project Gutenberg book: https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt
DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DEFAULT_INPUT = "data/input.txt"
# Fraction held out from the end of the corpus so compare.py can measure
# compression on genuinely unseen text. Must match HOLDOUT_FRAC in compare.py.
HOLDOUT_FRAC = 0.1


def ensure_corpus(path, url=DEFAULT_URL):
    """Download the training text to ``path`` if it isn't already present."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading corpus -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="path to training text")
    parser.add_argument("--vocab-size", type=int, default=1024, help="target vocabulary size (>=256)")
    parser.add_argument("--out", default="models/tok1024", help="output prefix for .model/.vocab")
    parser.add_argument("--holdout", type=float, default=HOLDOUT_FRAC,
                        help="fraction of the end of the corpus to exclude from training")
    parser.add_argument("--verbose", action="store_true", help="print each merge as it happens")
    args = parser.parse_args()

    # Only auto-download when the user relies on the default corpus path.
    if args.input == DEFAULT_INPUT:
        ensure_corpus(args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Train on everything except the held-out tail (kept unseen for compare.py).
    split = int(len(full_text) * (1 - args.holdout))
    text = full_text[:split]

    print(
        f"training on {args.input}: {len(text):,} of {len(full_text):,} chars "
        f"({len(text.encode('utf-8')):,} bytes, last {args.holdout:.0%} held out) "
        f"-> vocab_size={args.vocab_size}"
    )

    tok = RegexBPETokenizer()
    start = time.time()
    tok.train(text, args.vocab_size, verbose=args.verbose)
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_path = tok.save(args.out)

    print(f"trained {len(tok.merges)} merges in {elapsed:.1f}s")
    print(f"final vocabulary size: {len(tok.vocab):,}")
    print(f"saved -> {model_path} (+ {args.out}.vocab)")


if __name__ == "__main__":
    main()
