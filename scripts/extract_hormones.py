"""Extract hormone direction vectors from a (partially-)trained Unnatam checkpoint.

Examples
--------
    # extract from the 1B-token milestone, using gpt2 tokenizer
    python scripts/extract_hormones.py \
        --checkpoint runs/small_v1/ckpt_step10000.pt \
        --size small \
        --tokenizer gpt2 \
        --output runs/small_v1/hormones_step10000.npy

The output .npy is shaped (7, d_model) and can be passed back into the next
training stage via UnnatamConfig.hormone_vector_path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from unnatam.config import unnatam_medium, unnatam_small
from unnatam.hormones.definitions import HORMONE_NAMES
from unnatam.hormones.extract import extract_hormone_vectors, save_hormone_vectors
from unnatam.model import Unnatam
from unnatam.training.checkpoint import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to model .pt")
    p.add_argument("--size", choices=["small", "medium"], default="small")
    p.add_argument("--vocab_size", type=int, default=None,
                   help="override config vocab_size (defaults to tokenizer.vocab_size)")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--output", required=True, help="output .npy path")
    p.add_argument("--max_tokens", type=int, default=256, help="truncate prompts to N tokens")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = {"small": unnatam_small, "medium": unnatam_medium}[args.size]()
    cfg.vocab_size = args.vocab_size if args.vocab_size is not None else tok.vocab_size

    model = Unnatam(cfg)
    step = load_checkpoint(args.checkpoint, model, map_location=args.device)
    print(f"[extract] loaded {args.checkpoint} (step={step}) | d_model={cfg.d_model}")

    vectors = extract_hormone_vectors(
        model, tokenizer=tok, device=args.device, max_tokens=args.max_tokens
    )
    save_hormone_vectors(args.output, vectors)
    norms = np.linalg.norm(vectors, axis=1)
    print(f"[extract] wrote {args.output} shape={vectors.shape}")
    for name, n in zip(HORMONE_NAMES, norms):
        print(f"  {name:>4s}  L2={n:.4f}")


if __name__ == "__main__":
    main()
