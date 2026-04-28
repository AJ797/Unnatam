"""Pre-tokenize a streaming HuggingFace dataset into .bin shards.

Default: FineWeb-Edu sample-10BT, GPT-2 tokenizer (50k vocab — works without
auth and avoids HF Hub gating). Override with --dataset / --tokenizer.

Examples
--------
    # 1B tokens of FineWeb-Edu, 100M tokens per shard, GPT-2 BPE
    python scripts/prepare_data.py --output_dir data/fineweb --num_tokens 1_000_000_000

    # use Mistral tokenizer instead (32k, requires `transformers` + token in env)
    python scripts/prepare_data.py --tokenizer mistralai/Mistral-7B-v0.1 --output_dir data/fineweb_mistral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def stream_documents(dataset: str, name: str | None, split: str) -> Iterator[str]:
    from datasets import load_dataset

    kwargs: dict = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    ds = load_dataset(dataset, **kwargs)
    for doc in ds:
        text = doc.get("text") or doc.get("content")
        if text:
            yield text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT", help="dataset config name (subset)")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_tokens", type=int, default=1_000_000_000)
    p.add_argument("--shard_size", type=int, default=100_000_000)
    p.add_argument("--eot_token", type=int, default=None,
                   help="token id to append between docs; defaults to tokenizer.eos_token_id or 50256 for gpt2")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = args.eot_token if args.eot_token is not None else (tok.eos_token_id or 50256)
    vocab_size = tok.vocab_size
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    print(f"[prep] tokenizer={args.tokenizer} vocab_size={vocab_size} dtype={dtype.__name__} eot={eot}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard: list[int] = []
    total = 0
    shard_idx = 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        arr = np.array(shard, dtype=dtype)
        path = out / f"shard_{shard_idx:05d}.bin"
        arr.tofile(path)
        print(f"[prep] wrote {path} ({len(arr):,} tokens)")
        shard = []
        shard_idx += 1

    for text in stream_documents(args.dataset, args.name, args.split):
        ids = tok.encode(text)
        ids.append(eot)
        shard.extend(ids)
        total += len(ids)
        if len(shard) >= args.shard_size:
            flush()
        if total >= args.num_tokens:
            break

    flush()
    print(f"[prep] done. total_tokens={total:,} shards={shard_idx}")


if __name__ == "__main__":
    main()
