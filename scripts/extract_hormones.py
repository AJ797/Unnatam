"""Extract hormone direction vectors from a (partially-)trained Unnatam checkpoint.

Examples
--------
    # Typical: extract from the 1B-token milestone checkpoint
    python scripts/extract_hormones.py \\
        --ckpt runs/tiny_stage1/ckpt_milestone.pt \\
        --size tiny --out hormones.npy

    # HR-noext ablation: extract from a FRESHLY INITIALISED (untrained) model
    python scripts/extract_hormones.py \\
        --init --size tiny --out hormones_noext.npy

    # If the Stage-1 model used intracellular attention (for the Both variant)
    python scripts/extract_hormones.py \\
        --ckpt runs/ia_stage1/ckpt_milestone.pt \\
        --size tiny --intra_attn --out hormones_ia.npy

Output: float32 .npy of shape (n_hormones, d_model), L2-normalised rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from unnatam.config import unnatam_tiny, unnatam_small, unnatam_medium, unnatam_large
from unnatam.hormones.definitions import HORMONE_NAMES
from unnatam.hormones.extract import extract_hormone_vectors, save_hormone_vectors
from unnatam.model import Unnatam


# ---------------------------------------------------------------------------
# Tokenizer loader — tries tiktoken first (fast), falls back to transformers
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        class _TikTokenWrapper:
            def encode(self, text: str) -> list[int]:
                return enc.encode(text)

        return _TikTokenWrapper()
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")

        class _HFWrapper:
            def encode(self, text: str) -> list[int]:
                return tok.encode(text)

        return _HFWrapper()
    except ImportError:
        pass

    raise RuntimeError(
        "No tokenizer available. "
        "Install tiktoken (`pip install tiktoken`) "
        "or transformers (`pip install transformers`)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Checkpoint source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str,
                     help="path to checkpoint .pt to load weights from")
    src.add_argument("--init", action="store_true",
                     help="use a freshly-initialised (untrained) model — HR-noext ablation")

    # Model spec (must match the checkpoint's config)
    p.add_argument("--size", choices=["tiny", "small", "medium", "large"], default="tiny")
    p.add_argument("--vocab_size", type=int, default=50257,
                   help="vocab size used during training (must match checkpoint)")
    p.add_argument("--intra_attn", action="store_true",
                   help="set if checkpoint was trained with --intra_attn")

    # Extraction settings
    p.add_argument("--max_tokens", type=int, default=256,
                   help="max tokens per contrast-pair text (256 is plenty)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    p.add_argument("--out", type=str, required=True,
                   help="output .npy path")

    args = p.parse_args()

    # Build config — extraction hooks AttnLayer outputs, so no hormone module needed.
    cfg = {
        "tiny":   unnatam_tiny,
        "small":  unnatam_small,
        "medium": unnatam_medium,
        "large":  unnatam_large,
    }[args.size]()
    cfg.vocab_size = args.vocab_size
    cfg.use_intra_attn = args.intra_attn
    cfg.use_hormones = False  # don't add HormoneRouter — we're extracting, not injecting

    model = Unnatam(cfg)

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        # Handle both wrapped and unwrapped state dicts.
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[extract] WARNING: missing keys in state_dict: {missing[:5]} ...")
        print(f"[extract] loaded checkpoint: {args.ckpt}  (step={payload.get('step', '?')})")
    else:
        print("[extract] using freshly-initialised model (HR-noext ablation)")

    tokenizer = _load_tokenizer()

    print(f"[extract] running difference-of-means extraction on {args.device} …")
    vectors = extract_hormone_vectors(
        model, tokenizer,
        device=args.device,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_hormone_vectors(str(out_path), vectors)

    print(f"[extract] hormone bank written: {out_path}  shape={vectors.shape}")
    norms = np.linalg.norm(vectors, axis=-1)
    for name, norm in zip(HORMONE_NAMES, norms):
        print(f"    {name:>4s}  L2-norm={norm:.4f}  (should be ≈1.0)")


if __name__ == "__main__":
    main()
