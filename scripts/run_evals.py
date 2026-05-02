"""Run zero-shot benchmarks on a trained Unnatam checkpoint.

Outputs a JSON summary with per-benchmark accuracy.

Examples
--------
    # Single variant
    python scripts/run_evals.py \\
        --ckpt /workspace/runs/tiny_base/ckpt_final.pt \\
        --size tiny --out /workspace/runs/tiny_base/evals.json

    # All five variants in one go (driven by a shell loop)
    for v in tiny_base tiny_hr tiny_hr_rand tiny_hr_fixedgate tiny_hr_noext; do
        python scripts/run_evals.py \\
            --ckpt /workspace/runs/${v}/ckpt_final.pt \\
            --size tiny --out /workspace/runs/${v}/evals.json
    done

    # Quick smoke (subsamples each benchmark to 200 examples, ~5 min total)
    python scripts/run_evals.py --ckpt ... --size tiny --out ... --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from unnatam.benchmarks.eval import (
    eval_arc, eval_hellaswag, eval_lambada, eval_piqa,
)
from unnatam.config import unnatam_tiny, unnatam_small, unnatam_medium, unnatam_large
from unnatam.model import Unnatam


def _load_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")


def _load_model(ckpt_path: str, size: str, vocab_size: int, device: str,
                use_intra_attn: bool, use_hormones: bool, ia_stride: int) -> Unnatam:
    cfg = {
        "tiny":   unnatam_tiny,
        "small":  unnatam_small,
        "medium": unnatam_medium,
        "large":  unnatam_large,
    }[size]()
    cfg.vocab_size = vocab_size
    cfg.use_intra_attn = use_intra_attn
    cfg.use_hormones = use_hormones
    cfg.ia_stride = ia_stride

    model = Unnatam(cfg)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", choices=["tiny", "small", "medium", "large"], default="tiny")
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--use_intra_attn", action="store_true",
                   help="set if checkpoint was trained with --intra_attn")
    p.add_argument("--use_hormones", action="store_true",
                   help="set if checkpoint was trained with --hormones")
    p.add_argument("--ia_stride", type=int, default=32)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--quick", action="store_true",
                   help="subsample each benchmark to 200 examples (smoke test)")
    p.add_argument("--benchmarks", nargs="+",
                   default=["hellaswag", "lambada", "arc_easy", "arc_challenge", "piqa"])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"[evals] loading {args.ckpt} ({args.size})", flush=True)
    model = _load_model(args.ckpt, args.size, args.vocab_size, args.device,
                        args.use_intra_attn, args.use_hormones, args.ia_stride)
    tok = _load_tokenizer()
    n_examples = 200 if args.quick else None

    all_results = []
    for bench in args.benchmarks:
        print(f"\n[evals] === running {bench} ===", flush=True)
        if bench == "hellaswag":
            r = eval_hellaswag(model, tok, args.device, n_examples, args.max_length)
        elif bench == "lambada":
            r = eval_lambada(model, tok, args.device, n_examples, args.max_length)
        elif bench == "arc_easy":
            r = eval_arc(model, tok, args.device, "ARC-Easy", n_examples, args.max_length)
        elif bench == "arc_challenge":
            r = eval_arc(model, tok, args.device, "ARC-Challenge", n_examples, args.max_length)
        elif bench == "piqa":
            r = eval_piqa(model, tok, args.device, n_examples, args.max_length)
        else:
            print(f"  unknown benchmark: {bench}, skipping", flush=True)
            continue
        for res in r:
            print(f"  {res.name} {res.metric} = {res.value:.4f}  (n={res.n_examples})", flush=True)
            all_results.append(asdict(res))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "size": args.size,
        "use_intra_attn": args.use_intra_attn,
        "use_hormones": args.use_hormones,
        "results": all_results,
    }, indent=2))
    print(f"\n[evals] written {out_path}")


if __name__ == "__main__":
    main()
