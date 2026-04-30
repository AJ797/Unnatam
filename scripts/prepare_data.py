"""Pre-tokenize FineWeb-Edu (or any HF dataset) into flat binary .bin shards.

Tokenizer priority:
  1. tiktoken gpt2 (fastest, no HF auth needed)
  2. transformers AutoTokenizer (fallback)

Output layout (default):
  data/fineweb/train/shard_00000.bin
  data/fineweb/train/shard_00001.bin
  ...
  data/fineweb/val/shard_00000.bin    (held-out val slice)

Examples
--------
    # Full 10BT of FineWeb-Edu, GPT-2 BPE, 100M-token shards, 50M val
    python scripts/prepare_data.py \\
        --out /workspace/data/fineweb \\
        --num_tokens 10_000_000_000 --val_tokens 50_000_000

    # Quick 1BT test
    python scripts/prepare_data.py \\
        --out /workspace/data/fineweb_1bt \\
        --num_tokens 1_000_000_000 --val_tokens 10_000_000

    # Multiple workers for faster tokenization
    python scripts/prepare_data.py \\
        --out /workspace/data/fineweb --n_workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# ---------------------------------------------------------------------------
# Tokenizer — tiktoken is much faster than transformers for this task
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE = None


def _get_tokenizer(name: str = "gpt2"):
    """Return (tokenize_fn, vocab_size, eot_id)."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE

    try:
        import tiktoken
        enc = tiktoken.get_encoding(name)

        def tokenize(text: str) -> list[int]:
            return enc.encode_ordinary(text)

        result = (tokenize, enc.n_vocab, enc.eot_token)
        _TOKENIZER_CACHE = result
        return result
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)

        def tokenize(text: str) -> list[int]:
            return tok.encode(text)

        eot = tok.eos_token_id or 50256
        result = (tokenize, tok.vocab_size, eot)
        _TOKENIZER_CACHE = result
        return result
    except ImportError:
        pass

    raise RuntimeError(
        "No tokenizer found. Install tiktoken (`pip install tiktoken`) or "
        "transformers (`pip install transformers`)."
    )


# ---------------------------------------------------------------------------
# Data streaming
# ---------------------------------------------------------------------------

def _stream_texts(dataset: str, name: str | None, split: str) -> Iterator[str]:
    from datasets import load_dataset
    kwargs: dict = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    ds = load_dataset(dataset, **kwargs)
    for doc in ds:
        text = doc.get("text") or doc.get("content") or ""
        if text.strip():
            yield text


# ---------------------------------------------------------------------------
# Worker function (for multiprocessing pool)
# ---------------------------------------------------------------------------

def _tokenize_batch(batch: list[str]) -> list[int]:
    """Called in a subprocess — tokenize a list of texts, append EOT between each."""
    tokenize, _, eot = _get_tokenizer()
    tokens: list[int] = []
    for text in batch:
        tokens.extend(tokenize(text))
        tokens.append(eot)
    return tokens


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, dtype: np.dtype, prefix: str = "shard"):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.dtype = dtype
        self.prefix = prefix
        self._buf: list[int] = []
        self._idx = 0
        self.total = 0

    def write(self, tokens: list[int]) -> None:
        self._buf.extend(tokens)
        self.total += len(tokens)
        while len(self._buf) >= self.shard_size:
            chunk = self._buf[:self.shard_size]
            self._buf = self._buf[self.shard_size:]
            self._flush(chunk)

    def _flush(self, tokens: list[int]) -> None:
        path = self.out_dir / f"{self.prefix}_{self._idx:05d}.bin"
        np.array(tokens, dtype=self.dtype).tofile(path)
        print(f"  wrote {path}  ({len(tokens):,} tokens)", flush=True)
        self._idx += 1

    def close(self) -> None:
        if self._buf:
            self._flush(self._buf)
            self._buf = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset id")
    p.add_argument("--name", default="sample-10BT",
                   help="dataset config name (subset)")
    p.add_argument("--split", default="train",
                   help="dataset split to use")
    p.add_argument("--tokenizer", default="gpt2",
                   help="tiktoken encoding name OR HuggingFace tokenizer id")
    p.add_argument("--out", required=True,
                   help="output root directory (train/ and val/ subdirs created inside)")
    p.add_argument("--num_tokens", type=int, default=10_000_000_000,
                   help="total training tokens to produce")
    p.add_argument("--val_tokens", type=int, default=50_000_000,
                   help="held-out validation tokens (taken from the END of the stream)")
    p.add_argument("--shard_size", type=int, default=100_000_000,
                   help="tokens per shard file")
    p.add_argument("--batch_size", type=int, default=512,
                   help="documents per tokenization batch")
    p.add_argument("--n_workers", type=int, default=min(8, cpu_count()),
                   help="tokenization worker processes")
    args = p.parse_args()

    tokenize, vocab_size, eot = _get_tokenizer(args.tokenizer)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32

    print(f"[prep] tokenizer={args.tokenizer}  vocab_size={vocab_size}  "
          f"dtype={dtype.__name__}  eot={eot}")
    print(f"[prep] target: {args.num_tokens/1e9:.1f}B train + "
          f"{args.val_tokens/1e6:.0f}M val tokens")
    print(f"[prep] workers={args.n_workers}  shard_size={args.shard_size/1e6:.0f}M")

    out_root = Path(args.out)
    train_writer = ShardWriter(out_root / "train", args.shard_size, dtype)
    val_writer   = ShardWriter(out_root / "val",   args.shard_size, dtype)

    total_target = args.num_tokens + args.val_tokens
    doc_batch: list[str] = []

    def process_batch(batch: list[str]) -> None:
        if args.n_workers > 1:
            with Pool(args.n_workers) as pool:
                # Split batch among workers.
                chunk = max(1, len(batch) // args.n_workers)
                sub_batches = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]
                results = pool.map(_tokenize_batch, sub_batches)
            tokens: list[int] = []
            for r in results:
                tokens.extend(r)
        else:
            tokens = _tokenize_batch(batch)

        # Route: val_tokens come from the FRONT of the stream (makes it reproducible).
        if val_writer.total < args.val_tokens:
            need = args.val_tokens - val_writer.total
            val_chunk = tokens[:need]
            train_chunk = tokens[need:]
            val_writer.write(val_chunk)
            train_writer.write(train_chunk)
        else:
            train_writer.write(tokens)

    print(f"\n[prep] streaming {args.dataset}/{args.name} …")
    for text in _stream_texts(args.dataset, args.name, args.split):
        doc_batch.append(text)
        if len(doc_batch) >= args.batch_size:
            process_batch(doc_batch)
            doc_batch = []
            done = train_writer.total + val_writer.total
            print(f"[prep] {done/1e9:.3f}B / {total_target/1e9:.1f}B tokens", flush=True)
            if done >= total_target:
                break

    if doc_batch and train_writer.total + val_writer.total < total_target:
        process_batch(doc_batch)

    train_writer.close()
    val_writer.close()

    print(f"\n[prep] done.")
    print(f"  train: {train_writer.total/1e9:.3f}B tokens  ({train_writer._idx} shards) → {out_root/'train'}")
    print(f"  val:   {val_writer.total/1e6:.1f}M tokens  ({val_writer._idx} shards) → {out_root/'val'}")


if __name__ == "__main__":
    main()
