"""Inference-time activation-addition steering on pretrained Mamba-1.4B.

Uses the hormone bank extracted by scripts/extract_hormones_mamba14b.py.
For each prompt and each hormone i we install forward hooks on the same layers
used for extraction; each hook adds alpha * v_i to the residual stream at every
token position, then we generate a continuation.

Runs locally on a 6GB 4060 in fp16. ~15-30 seconds per generation.

Requirements
------------
    pip install torch transformers mamba-ssm einops

Usage
-----
    py scripts/steering_test_mamba14b.py \
        --model_id state-spaces/mamba-1.4b-hf \
        --hormones_npy mamba14b_hormones.npy \
        --meta mamba14b_hormones.meta.json \
        --alphas 0 4 8 \
        --max_new_tokens 80 \
        --temperature 0.7 \
        --seed 0 \
        --out mamba14b_steering.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unnatam.hormones.definitions import HORMONE_NAMES  # noqa: E402


HORMONE_INTENT = {
    "ADR":  "urgency",
    "CDO":  "curiosity",
    "LCO":  "skepticism",
    "NRA":  "persistence",
    "OXY":  "warmth",
    "SRO":  "equanimity",
    "SELF": "self-coherence",
}

PROMPTS = [
    "The team gathered around the table. She looked up and said,",
    "There's a strange pattern in the data. I'd like your advice on",
    "The dog padded across the kitchen floor and",
    "He opened the letter and read the first line, which said",
    "In the article they described how the new system",
]


def install_addition_hooks(layers, extraction_layers, direction: torch.Tensor,
                            alpha: float, device: str):
    """Install forward hooks that add alpha * direction to the residual at each
    extraction layer, at every token position.

    direction: (d_model,) unit-norm tensor on `device`.
    """
    handles = []
    add = (alpha * direction).to(device)

    def make_hook():
        def hook(_mod, _inp, output):
            # output is either a Tensor (h) or a tuple whose [0] is h
            if isinstance(output, tuple):
                h = output[0]
                h = h + add.to(h.dtype)
                return (h,) + tuple(output[1:])
            else:
                return output + add.to(output.dtype)
        return hook

    for li in extraction_layers:
        handles.append(layers[li].register_forward_hook(make_hook()))
    return handles


@torch.no_grad()
def sample(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
           device: str) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits = model(ids).logits                          # (1, T, V)
        next_logits = logits[0, -1, :] / max(1e-6, temperature)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id[None, :]], dim=1)
        if ids.shape[1] >= 1024:
            break
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_id", default="state-spaces/mamba-1.4b-hf")
    p.add_argument("--hormones_npy", required=True)
    p.add_argument("--meta", required=True, help="metadata .json from extraction")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 4.0, 8.0],
                   help="steering strengths to test. 0 is baseline (no steering).")
    p.add_argument("--hormones", nargs="+", default=None,
                   help="which hormones to test (default: all 7)")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="custom prompts (default: 5 built-in)")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_id}  dtype={dtype}  device={args.device}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(args.device).eval()
    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    layers = backbone.layers

    meta = json.loads(Path(args.meta).read_text())
    extraction_layers = meta["extraction_layers"]
    print(f"[load] extraction_layers={extraction_layers}")

    bank = np.load(args.hormones_npy)                     # (n_hormones, d_model)
    print(f"[load] hormone bank shape={bank.shape}")

    hormones = args.hormones or meta["hormone_names"]
    prompts = args.prompts or PROMPTS

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    lines: list[str] = []
    def emit(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    emit(f"=== steering: model={args.model_id}, alphas={args.alphas}, T={args.temperature}, seed={args.seed} ===")

    for prompt in prompts:
        emit(f"\n─── PROMPT ─── {prompt}")

        # Baseline (no hooks, alpha=0)
        torch.manual_seed(args.seed)
        base_out = sample(model, tok, prompt, args.max_new_tokens, args.temperature, args.device)
        emit(f"\n[baseline]\n  {base_out}")

        for name in hormones:
            if name not in meta["hormone_names"]:
                emit(f"[skip] unknown hormone {name}")
                continue
            idx = meta["hormone_names"].index(name)
            direction = torch.from_numpy(bank[idx])

            for alpha in args.alphas:
                if alpha == 0.0:
                    continue  # baseline already printed
                handles = install_addition_hooks(layers, extraction_layers, direction, alpha, args.device)
                try:
                    torch.manual_seed(args.seed)
                    steered = sample(model, tok, prompt, args.max_new_tokens, args.temperature, args.device)
                finally:
                    for h in handles:
                        h.remove()
                emit(f"\n[{name} ({HORMONE_INTENT.get(name, '?')}) alpha={alpha:+.1f}]\n  {steered}")

        emit("")

    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
