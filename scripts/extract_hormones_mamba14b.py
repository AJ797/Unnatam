"""Extract hormone direction vectors from a PRETRAINED Mamba-1.4B model.

Motivation
----------
Our trained 125M routing develops semantic specialisation (Section 6.3, Section 7)
but its effective injection magnitude sits inside the homeostatic envelope, so we
cannot demonstrate behavioural steering by turning up its dial. To support the
"hormone" naming as a functional label rather than an ornament, we apply the same
document-style contrast-pair extraction protocol to a much larger pretrained model
(state-spaces/mamba-1.4b-hf, 1.4B params) and use activation-addition steering to
show that the extracted directions produce hormone-labelled behavioural shifts.

This mirrors the methodology of Turner et al. 2023 / Rimsky et al. 2024 for
representation steering, applied through OUR contrast-pair construction.

Runs locally on a 6GB 4060 in fp16 (~20 min total for all 7 hormones × 6 pairs).

Requirements
------------
    pip install torch transformers mamba-ssm einops

Usage
-----
    py scripts/extract_hormones_mamba14b.py \
        --model_id state-spaces/mamba-1.4b-hf \
        --extraction_layers 24 25 26 27 28 29 30 31 \
        --out mamba14b_hormones.npy \
        --meta mamba14b_hormones.meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unnatam.hormones.definitions import HORMONE_CONTRASTS, HORMONE_NAMES  # noqa: E402


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    """Load a HuggingFace Mamba-family model. Returns (model, tokenizer, n_layers, d_model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {model_id}  dtype={dtype}  device={device}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    # Extract layer count and hidden dim
    # Mamba-hf structure: model.backbone.layers is a ModuleList of MambaBlock
    # Fallback: try common attributes
    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError(f"Could not find backbone in {model_id}")
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError(f"Could not find backbone.layers in {model_id}")
    n_layers = len(layers)
    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model
    print(f"[load] n_layers={n_layers}  d_model={d_model}")
    return model, tok, n_layers, d_model, layers


@torch.no_grad()
def extract(
    model,
    tokenizer,
    layers,                       # backbone.layers ModuleList
    extraction_layers: list[int], # which layer indices to hook (0-based)
    contrasts: dict,
    device: str,
    max_tokens: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """Return (n_hormones, d_model) L2-normalised hormone direction bank.

    For each contrast pair (d+, d-):
      - forward both through model
      - collect residual-stream activation OUT of each extraction_layer
      - mean across selected layers and across token positions (excluding padding)
      - compute pos - neg
    Average the pair-differences per hormone, then L2-normalise.
    """
    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model

    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, output):
        # HF Mamba layer output is a tuple (hidden_states, ...) or a plain Tensor
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h)

    handles = []
    for li in extraction_layers:
        handles.append(layers[li].register_forward_hook(hook))

    def collect_mean(text: str) -> torch.Tensor:
        captured.clear()
        ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                               max_length=max_tokens).to(device)
        _ = model(ids)
        stacked = torch.stack(captured, dim=0)              # (n_layers_hooked, 1, T, d_model)
        return stacked.mean(dim=(0, 2)).squeeze(0).float().cpu()  # (d_model,)

    try:
        names = list(contrasts.keys())
        vectors = np.zeros((len(names), d_model), dtype=np.float32)
        for hi, name in enumerate(names):
            pairs = contrasts[name]
            diffs = []
            for pi, pair in enumerate(pairs):
                pos_v = collect_mean(pair.positive)
                neg_v = collect_mean(pair.negative)
                diffs.append((pos_v - neg_v).numpy())
                if verbose:
                    print(f"  [{name} pair {pi + 1}/{len(pairs)}] "
                          f"||pos-neg||={np.linalg.norm(diffs[-1]):.4f}")
            mean_diff = np.mean(diffs, axis=0)
            norm = np.linalg.norm(mean_diff)
            v = mean_diff / max(norm, 1e-8)
            vectors[hi] = v
            if verbose:
                print(f"[{name}] mean-diff norm={norm:.4f}, normalised to unit length")
    finally:
        for h in handles:
            h.remove()

    return vectors


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_id", default="state-spaces/mamba-1.4b-hf")
    p.add_argument("--extraction_layers", type=int, nargs="+", default=None,
                   help="which layer indices to hook. Default: middle 8 layers.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--out", required=True, help="path to .npy hormone bank")
    p.add_argument("--meta", default=None, help="optional metadata .json path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    model, tok, n_layers, d_model, layers = _load_model(args.model_id, args.device, dtype)

    # Default extraction layers: the middle 8 (approx upper-mid, per Zou 2023)
    if args.extraction_layers is None:
        mid = n_layers // 2
        args.extraction_layers = list(range(max(0, mid - 4), min(n_layers, mid + 4)))
    print(f"[extract] extraction_layers={args.extraction_layers}")

    print(f"[extract] running {sum(len(p) for p in HORMONE_CONTRASTS.values())} pairs "
          f"across {len(HORMONE_CONTRASTS)} hormones…")
    vectors = extract(model, tok, layers, args.extraction_layers,
                      HORMONE_CONTRASTS, args.device, args.max_tokens)

    np.save(args.out, vectors)
    print(f"\n[extract] wrote {args.out}  shape={vectors.shape}")
    print(f"[extract] per-row norms: {np.linalg.norm(vectors, axis=-1).round(4).tolist()}")

    if args.meta:
        meta = {
            "model_id": args.model_id,
            "n_layers": n_layers,
            "d_model": d_model,
            "extraction_layers": args.extraction_layers,
            "max_tokens": args.max_tokens,
            "hormone_names": list(HORMONE_CONTRASTS.keys()),
        }
        Path(args.meta).write_text(json.dumps(meta, indent=2))
        print(f"[extract] wrote {args.meta}")


if __name__ == "__main__":
    main()
