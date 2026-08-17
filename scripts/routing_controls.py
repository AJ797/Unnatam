"""Reviewer f2c6 requested controls for the routing-specialisation claim.

For a trained HR checkpoint we compute, on a held-out FineWeb-Edu-like corpus
of N documents per genre:

  1) Per-document mean routing distributions at the engaged upper attention
     layer, then per-genre mean and std of max pairwise L1 across the six
     genres. Answers: is the specialisation consistent across documents, or
     driven by one selected passage?

  2) Document-level permutation null: pool the per-document mean distributions
     across genres, permute the document->genre assignment K times, compute
     max pairwise L1 of the K partitions. Reports null median / 95th / 99th
     percentile and the observed L1 with an empirical p-value.

  3) UNTRAINED-ROUTER baseline: instantiate a fresh Unnatam model (zero
     training), do the same analysis. Answers: does the observed structure
     require a trained router, or would a random-init router produce
     genre-discriminative outputs from residual-stream input differences alone?

  4) Effective residual perturbation magnitude ||Delta h||_2 / ||h||_2 at each
     attention layer, averaged over documents and token positions. Answers
     reviewer f2c6's request for a real perturbation measurement rather than
     just mu = |g| * mean(m).

Runs on the 4060 in fp16. Reads documents from a text file (one document per
line, or one document per --sep block). Provide your own held-out corpus with
6 genre files named news.txt, fiction.txt, code.txt, dialog.txt, tech.txt,
urgent.txt in --corpus_dir, each with >= --n_per_genre documents.

Usage:
    py scripts/routing_controls.py \
        --ckpt runs/tiny_hr_forced/ckpt_final.pt \
        --corpus_dir data/heldout_genres \
        --n_per_genre 20 \
        --layer 11 \
        --n_permutations 2000 \
        --out runs/tiny_hr_forced/routing_controls.json

For the untrained baseline (no --ckpt):
    py scripts/routing_controls.py \
        --untrained \
        --corpus_dir data/heldout_genres \
        --n_per_genre 20 \
        --layer 11 \
        --out runs/untrained/routing_controls.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unnatam.config import unnatam_tiny  # noqa: E402
from unnatam.model.unnatam import Unnatam  # noqa: E402


GENRE_NAMES = ["news", "fiction", "code", "dialog", "tech", "urgent"]


def load_docs(corpus_dir: Path, n_per_genre: int, min_tokens: int = 32):
    """Return dict genre -> list of doc strings, at most n_per_genre each.
    Assumes one document per line in each genre's file."""
    out = {}
    for g in GENRE_NAMES:
        path = corpus_dir / f"{g}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}. Expected one doc per line.")
        docs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        docs = [d for d in docs if len(d) >= min_tokens * 3]  # crude char-len filter
        if len(docs) < n_per_genre:
            raise ValueError(f"only {len(docs)} docs in {path.name}, need {n_per_genre}")
        out[g] = docs[:n_per_genre]
    return out


def max_pairwise_l1(dists: np.ndarray) -> float:
    """dists: (n_groups, n_hormones), each row sums to 1."""
    return float(np.abs(dists[:, None, :] - dists[None, :, :]).sum(-1).max())


@torch.no_grad()
def collect_layer_routing(model, tokenizer, layer_idx: int, docs: dict,
                          device: str) -> dict:
    """For each genre and each document, return a per-document mean routing
    distribution over positions at the given attention layer, PLUS the
    ||Delta h|| / ||h|| ratio per layer per document."""
    layer = model.layers[layer_idx]
    if not (hasattr(layer, "hormone") and layer.hormone is not None):
        raise ValueError(f"Layer {layer_idx} has no hormone module")

    per_layer_hormone_layers = [
        i for i, ly in enumerate(model.layers)
        if hasattr(ly, "hormone") and ly.hormone is not None
    ]

    per_doc_routing: dict[str, list[np.ndarray]] = {g: [] for g in docs}
    per_doc_delta_ratios: dict[int, list[float]] = {i: [] for i in per_layer_hormone_layers}

    for g, dlist in docs.items():
        for d in dlist:
            captured_w: dict = {}
            captured_h_norms: dict = {}
            captured_delta_norms: dict = {}

            def make_hook(li):
                def fn(mod, inp, out):
                    h = inp[0]                                       # (B, T, d_model)
                    with torch.no_grad():
                        w = torch.softmax(mod.router(h), dim=-1)      # (B, T, n_h)
                        delta = out                                   # (B, T, d_model) -- the shift returned
                        h_norm = h.float().norm(dim=-1).mean().item()  # scalar
                        d_norm = delta.float().norm(dim=-1).mean().item()
                    captured_w[li] = w[0].detach().cpu().numpy()      # (T, n_h)
                    captured_h_norms[li] = h_norm
                    captured_delta_norms[li] = d_norm
                return fn

            handles = []
            for li in per_layer_hormone_layers:
                handles.append(model.layers[li].hormone.register_forward_hook(make_hook(li)))

            try:
                ids = torch.tensor([tokenizer.encode(d)[:512]], device=device, dtype=torch.long)
                _ = model(ids)
            finally:
                for h in handles:
                    h.remove()

            # per-doc mean routing at chosen layer
            rows = captured_w[layer_idx]                              # (T, n_h)
            per_doc_routing[g].append(rows.mean(axis=0))

            # per-layer delta / h ratio for this doc
            for li in per_layer_hormone_layers:
                ratio = captured_delta_norms[li] / max(1e-8, captured_h_norms[li])
                per_doc_delta_ratios[li].append(ratio)

    return per_doc_routing, per_doc_delta_ratios


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="trained-model checkpoint")
    src.add_argument("--untrained", action="store_true",
                     help="use a fresh untrained Unnatam model (baseline)")

    p.add_argument("--corpus_dir", required=True, type=Path,
                   help="directory with news.txt, fiction.txt, code.txt, dialog.txt, tech.txt, urgent.txt "
                        "(each file = one doc per line)")
    p.add_argument("--n_per_genre", type=int, default=20)
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="optional .json output")
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    class _TikTok:
        def encode(self, text):
            return enc.encode_ordinary(text)

    tok = _TikTok()

    cfg = unnatam_tiny()
    cfg.vocab_size = 50257
    cfg.use_hormones = True
    m = Unnatam(cfg).to(args.device).eval()

    if not args.untrained:
        sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        miss, _ = m.load_state_dict(sd, strict=False)
        if miss:
            print(f"[warn] {len(miss)} missing keys (first 3): {miss[:3]}")
    else:
        print("[untrained] using fresh init; no checkpoint loaded")

    print(f"[load] loading corpus from {args.corpus_dir}...")
    docs = load_docs(args.corpus_dir, args.n_per_genre)
    for g, dlist in docs.items():
        print(f"  {g}: {len(dlist)} docs")

    print(f"[capture] running {sum(len(v) for v in docs.values())} forward passes at layer {args.layer}...")
    per_doc_routing, per_doc_delta_ratios = collect_layer_routing(
        m, tok, args.layer, docs, args.device
    )

    # ----- per-genre mean and std of max pairwise L1 across documents -----
    print("\n=== Per-genre routing consistency ===")
    genre_means = np.stack([np.stack(per_doc_routing[g]).mean(axis=0) for g in GENRE_NAMES])
    observed_l1 = max_pairwise_l1(genre_means)
    print(f"  observed max pairwise L1 (per-genre means): {observed_l1:.4f}")

    # per-document max pairwise L1 (compare each doc against genre means)
    all_docs_flat = []
    for g in GENRE_NAMES:
        for row in per_doc_routing[g]:
            all_docs_flat.append((g, row))
    print(f"  total docs pooled: {len(all_docs_flat)}")

    # ----- document-level permutation null -----
    rng = np.random.default_rng(args.seed)
    n_docs = len(all_docs_flat)
    rows_pool = np.stack([r for _, r in all_docs_flat])          # (n_docs, n_h)
    genre_sizes = [len(per_doc_routing[g]) for g in GENRE_NAMES]

    null_l1s = np.zeros(args.n_permutations, dtype=np.float32)
    for i in range(args.n_permutations):
        perm = rng.permutation(n_docs)
        chunks = []
        cursor = 0
        for sz in genre_sizes:
            chunks.append(rows_pool[perm[cursor:cursor + sz]].mean(axis=0))
            cursor += sz
        null_l1s[i] = max_pairwise_l1(np.stack(chunks))

    q50 = float(np.percentile(null_l1s, 50))
    q95 = float(np.percentile(null_l1s, 95))
    q99 = float(np.percentile(null_l1s, 99))
    p_value = float(np.mean(null_l1s >= observed_l1))

    print(f"\n=== Document-level permutation null (N={args.n_permutations}) ===")
    print(f"  null median:    {q50:.4f}")
    print(f"  null 95%ile:    {q95:.4f}")
    print(f"  null 99%ile:    {q99:.4f}")
    print(f"  observed:       {observed_l1:.4f}")
    print(f"  empirical p:    {p_value:.4f}")

    # ----- per-layer ||Delta h|| / ||h|| -----
    print("\n=== Per-layer effective perturbation magnitude ||Delta h|| / ||h|| ===")
    per_layer_stats = {}
    for li, ratios in per_doc_delta_ratios.items():
        arr = np.array(ratios)
        per_layer_stats[li] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "median": float(np.median(arr)),
            "n_docs": int(arr.size),
        }
        print(f"  layer {li}: mean={arr.mean():.5f}  std={arr.std():.5f}  median={np.median(arr):.5f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt if not args.untrained else "UNTRAINED",
            "layer": args.layer,
            "n_per_genre": args.n_per_genre,
            "n_permutations": args.n_permutations,
            "genre_sizes": dict(zip(GENRE_NAMES, genre_sizes)),
            "observed_l1": observed_l1,
            "null_median": q50,
            "null_95": q95,
            "null_99": q99,
            "empirical_p": p_value,
            "genre_means": {g: genre_means[i].tolist() for i, g in enumerate(GENRE_NAMES)},
            "per_layer_delta_over_h": per_layer_stats,
        }, indent=2))
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
