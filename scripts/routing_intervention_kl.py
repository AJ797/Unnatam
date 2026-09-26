"""Inference-time routing interventions (Action-Editor-requested, TMLR 10166).

The Action Editor's most-direct test of whether the model *uses* the routing
structure: at inference, replace the learned per-token routing with a
content-free alternative and measure how much the model's predictions move.

Three interventions, all applied to a trained checkpoint with no retraining:

  * uniform   -- replace the softmax routing weights with the uniform
                 distribution 1/n_hormones at every position. The injection
                 becomes a constant alpha*gate*mean_i(m_i v_i); the per-token
                 structure is removed but the channel still contributes.
  * shuffled  -- keep each position's routing weights but randomly permute
                 which position gets which weight vector (within the sequence).
                 Marginal routing distribution is preserved; the token->routing
                 correspondence is destroyed.
  * zerogate  -- set the output gate to 0, i.e. the unmodified backbone. The
                 routing channel is fully removed.

For each we report, on a held-out corpus:

  * validation NLL / PPL under the intervention (vs. the real-routing PPL),
  * mean per-token KL( P_real || P_intervened ) between next-token predictive
    distributions (nats), and the reverse KL,
  * top-1 flip rate: fraction of positions whose argmax next-token prediction
    changes under the intervention.

A near-zero KL / flip-rate is itself the finding: it is direct evidence that,
under the tested conditions, the model's predictions barely depend on the
routing structure -- consistent with the channel contributing very little
(the interpretation the Action Editor asked us to make explicit), not with the
model actively exploiting a large absorbed injection.

Usage (HR-forced; alpha=3 reproduces its true forced injection):
    py scripts/routing_intervention_kl.py \
        --ckpt runs/tiny_hr_forced/ckpt_final.pt --alpha 3.0 \
        --corpus_dir data/heldout_genres --n_per_genre 20 \
        --out runs/tiny_hr_forced/routing_intervention_kl.json

For an unforced variant, pass its own alpha (1.0) and checkpoint.
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

from unnatam.config import unnatam_tiny  # noqa: E402
from unnatam.model.unnatam import Unnatam  # noqa: E402

GENRE_NAMES = ["news", "fiction", "code", "dialog", "tech", "urgent"]


def load_docs(corpus_dir: Path, n_per_genre: int, min_tokens: int = 32):
    out = {}
    for g in GENRE_NAMES:
        path = corpus_dir / f"{g}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}. Expected one doc per line.")
        docs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        docs = [d for d in docs if len(d) >= min_tokens * 3]
        if len(docs) < n_per_genre:
            raise ValueError(f"only {len(docs)} docs in {path.name}, need {n_per_genre}")
        out[g] = docs[:n_per_genre]
    return out


def make_intervention_hook(mode: str, rng: np.random.Generator):
    """Return a forward hook for a HormoneRouter that overrides its output.

    The hook recomputes the injection from the captured input h so it can apply
    the exact same alpha/gate/magnitudes the module would, then swaps in the
    intervened routing weights."""
    def hook(module, inputs, output):
        h = inputs[0]                                            # (B, T, d_model)
        hormone_vectors = inputs[1]                              # (n_h, d_model)
        scaled_v = hormone_vectors * module.magnitudes.unsqueeze(-1)   # (n_h, d)
        coeff = module.alpha * module.gate                       # scalar tensor

        if mode == "zerogate":
            return torch.zeros_like(output)

        if mode == "uniform":
            n_h = scaled_v.shape[0]
            mean_v = scaled_v.mean(dim=0)                        # (d,)
            return coeff * mean_v.expand_as(output)

        if mode == "shuffled":
            logits = module.router(h)                            # (B, T, n_h)
            weights = F.softmax(logits, dim=-1)
            B, T, n_h = weights.shape
            perm = torch.from_numpy(rng.permutation(T)).to(weights.device)
            shuffled = weights[:, perm, :]                       # permute positions
            return coeff * (shuffled @ scaled_v)

        raise ValueError(f"unknown mode {mode}")
    return hook


@torch.no_grad()
def logits_for_doc(model, ids):
    out = model(ids)
    return out.logits if hasattr(out, "logits") else out         # (B, T, vocab)


@torch.no_grad()
def run(model, tok, docs, device, modes):
    hormone_layers = [
        i for i, ly in enumerate(model.layers)
        if hasattr(ly, "hormone") and ly.hormone is not None
    ]
    if not hormone_layers:
        raise ValueError("model has no hormone layers")

    rng = np.random.default_rng(0)

    # accumulators
    tot_tokens = 0
    real_nll = 0.0
    stats = {m: dict(nll=0.0, kl_fwd=0.0, kl_rev=0.0, flips=0, tokens=0) for m in modes}

    for g, dlist in docs.items():
        for d in dlist:
            ids = torch.tensor([tok.encode(d)[:512]], device=device, dtype=torch.long)
            if ids.shape[1] < 2:
                continue
            targets = ids[:, 1:]                                 # (1, T-1)

            real_logits = logits_for_doc(model, ids)[:, :-1, :]  # (1, T-1, V)
            real_logp = F.log_softmax(real_logits.float(), dim=-1)
            real_p = real_logp.exp()
            n_tok = targets.numel()
            tot_tokens += n_tok
            real_nll += F.nll_loss(
                real_logp.reshape(-1, real_logp.shape[-1]),
                targets.reshape(-1), reduction="sum").item()
            real_argmax = real_logits.argmax(dim=-1)             # (1, T-1)

            for m in modes:
                hook = make_intervention_hook(m, rng)
                handles = [model.layers[i].hormone.register_forward_hook(hook)
                           for i in hormone_layers]
                try:
                    iv_logits = logits_for_doc(model, ids)[:, :-1, :]
                finally:
                    for hd in handles:
                        hd.remove()

                iv_logp = F.log_softmax(iv_logits.float(), dim=-1)
                # KL(real || intervened) = sum real_p * (real_logp - iv_logp)
                kl_fwd = (real_p * (real_logp - iv_logp)).sum(-1)          # (1, T-1)
                iv_p = iv_logp.exp()
                kl_rev = (iv_p * (iv_logp - real_logp)).sum(-1)
                stats[m]["kl_fwd"] += kl_fwd.sum().item()
                stats[m]["kl_rev"] += kl_rev.sum().item()
                stats[m]["nll"] += F.nll_loss(
                    iv_logp.reshape(-1, iv_logp.shape[-1]),
                    targets.reshape(-1), reduction="sum").item()
                stats[m]["flips"] += (iv_logits.argmax(dim=-1) != real_argmax).sum().item()
                stats[m]["tokens"] += n_tok

    real_ppl = float(np.exp(real_nll / tot_tokens))
    result = {
        "n_tokens": tot_tokens,
        "real_routing": {"nll": real_nll / tot_tokens, "ppl": real_ppl},
        "interventions": {},
    }
    for m in modes:
        s = stats[m]
        t = s["tokens"]
        result["interventions"][m] = {
            "ppl": float(np.exp(s["nll"] / t)),
            "delta_ppl": float(np.exp(s["nll"] / t) - real_ppl),
            "mean_kl_fwd_nats": s["kl_fwd"] / t,
            "mean_kl_rev_nats": s["kl_rev"] / t,
            "top1_flip_rate": s["flips"] / t,
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--alpha", type=float, default=3.0,
                   help="hormone_alpha at construction (HR-forced=3.0, unforced=1.0)")
    p.add_argument("--corpus_dir", required=True, type=Path)
    p.add_argument("--n_per_genre", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    class _Tok:
        def encode(self, text):
            return enc.encode_ordinary(text)

    tok = _Tok()

    cfg = unnatam_tiny()
    cfg.vocab_size = 50257
    cfg.use_hormones = True
    cfg.hormone_alpha = args.alpha
    model = Unnatam(cfg).to(args.device).eval()

    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss, _ = model.load_state_dict(sd, strict=False)
    if miss:
        print(f"[warn] {len(miss)} missing keys (first 3): {miss[:3]}")

    docs = load_docs(args.corpus_dir, args.n_per_genre)
    print(f"[run] {sum(len(v) for v in docs.values())} docs, alpha={args.alpha}")

    modes = ["uniform", "shuffled", "zerogate"]
    result = run(model, tok, docs, args.device, modes)

    print(f"\nreal-routing PPL: {result['real_routing']['ppl']:.4f}  "
          f"({result['n_tokens']} tokens)")
    for m in modes:
        r = result["interventions"][m]
        print(f"  {m:9s}  PPL={r['ppl']:.4f} (Δ{r['delta_ppl']:+.4f})  "
              f"KL(real||iv)={r['mean_kl_fwd_nats']:.4e} nats  "
              f"flip={r['top1_flip_rate']:.4%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "alpha": args.alpha, **result}, indent=2))
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
