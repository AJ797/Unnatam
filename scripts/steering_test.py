"""Inference-time behavioural-steering test.

Reviewer tkcT #2: the name "hormone" implies behavioural steering, but the
paper only measures loss and router statistics — it never checks whether
forcing a direction actually changes generated text.

This script tests exactly that. For a trained HR checkpoint:
  1. Baseline generation: no modification. Sample continuations from prompts.
  2. Per-hormone forced generation: override the router at inference time so
     that w(h) is one-hot on hormone i, boost the gate by a fixed factor,
     sample continuations from the same prompts.
  3. Print baseline vs each forced variant, so a reader can qualitatively
     judge whether the "hormone" naming corresponds to any observable shift.

The steering intervention is applied via forward hooks on each HormoneRouter:
  - Replace router logits with a one-hot mask (very high logit on target i)
  - Multiply the module's gate by --gate_boost so the injection is felt

Runs on CPU or a small GPU (~1-2 min per prompt on the 4060).

Usage:
    py scripts/steering_test.py \
        --ckpt runs/tiny_hr/ckpt_final.pt \
        --gate_boost 5.0 \
        --max_new_tokens 60 \
        --temperature 0.8 \
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unnatam.config import unnatam_medium  # noqa: E402
from unnatam.model.unnatam import Unnatam  # noqa: E402


HORMONE_LABELS = ["ADR", "CDO", "LCO", "NRA", "OXY", "SRO", "SELF"]
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
    "The dog padded across the kitchen floor and",
    "In the article they described how the new system",
    "He opened the letter and read the first line, which said",
]


def install_steering_hooks(model, target_hormone_idx: int, gate_boost: float):
    """Install forward hooks on every HormoneRouter so that:
      - router logits are replaced by a large one-hot value on target_hormone_idx
      - the gate value used in the shift is multiplied by gate_boost.

    Returns a list of handles for later removal.
    """
    handles = []

    # The HormoneRouter forward does:
    #   logits = router(h);  w = softmax(logits);
    #   scaled_v = hormone_vectors * magnitudes;  shift = w @ scaled_v
    #   return alpha * gate * shift
    # To force one-hot routing on target i we replace `router(h)` output.

    def make_router_hook(idx):
        def hook(mod, inp, out):  # noqa: ARG001
            # out is (B, T, n_hormones). Replace with strong one-hot on idx.
            logits = torch.full_like(out, -1e4)
            logits[..., idx] = 1e4
            return logits
        return hook

    for layer in model.layers:
        if hasattr(layer, "hormone") and layer.hormone is not None:
            h_router = layer.hormone.router.register_forward_hook(make_router_hook(target_hormone_idx))
            handles.append(h_router)
            # Boost the gate in-place (revert on teardown)
            layer.hormone.gate.data.mul_(gate_boost)

    return handles


def uninstall_steering_hooks(model, handles, gate_boost: float):
    for h in handles:
        h.remove()
    for layer in model.layers:
        if hasattr(layer, "hormone") and layer.hormone is not None:
            layer.hormone.gate.data.div_(gate_boost)


@torch.no_grad()
def sample(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
           device: str) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(ids)                                 # (1, T, V)
        next_logits = logits[0, -1, :] / max(1e-6, temperature)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id[None, :]], dim=1)
        if ids.shape[1] >= 1024:
            break
    return tokenizer.decode(ids[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gate_boost", type=float, default=5.0,
                   help="multiplier on the router gate for forced-hormone runs")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hormones", nargs="+", default=None,
                   help="which hormones to test (default: all 7)")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="custom prompts (default: 4 built-in)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="optional text output path")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    class _TikTok:
        def encode(self, text):
            return enc.encode_ordinary(text)
        def decode(self, ids):
            return enc.decode(ids)

    tok = _TikTok()

    cfg = unnatam_medium()
    cfg.vocab_size = 50257
    cfg.use_hormones = True
    m = Unnatam(cfg).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    missing, _ = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 3): {missing[:3]}")

    hormones = args.hormones or HORMONE_LABELS
    prompts = args.prompts or PROMPTS
    lines = []

    def emit(msg: str) -> None:
        print(msg)
        lines.append(msg)

    emit(f"=== steering test: ckpt={args.ckpt}, gate_boost={args.gate_boost}, T={args.temperature}, seed={args.seed} ===\n")

    for prompt in prompts:
        emit(f"\n─── PROMPT ─── {prompt}")
        # Baseline (no hooks)
        torch.manual_seed(args.seed)
        base_out = sample(m, tok, prompt, args.max_new_tokens, args.temperature, args.device)
        emit(f"\n[baseline]\n  {base_out}")

        for name in hormones:
            if name not in HORMONE_LABELS:
                emit(f"[skip] unknown hormone {name}")
                continue
            idx = HORMONE_LABELS.index(name)
            handles = install_steering_hooks(m, idx, args.gate_boost)
            try:
                torch.manual_seed(args.seed)
                steered = sample(m, tok, prompt, args.max_new_tokens, args.temperature, args.device)
            finally:
                uninstall_steering_hooks(m, handles, args.gate_boost)
            emit(f"\n[force {name}={HORMONE_INTENT[name]}, gate×{args.gate_boost}]\n  {steered}")

        emit("")

    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
