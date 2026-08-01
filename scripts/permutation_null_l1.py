"""Permutation-based null distribution for the routing-specialisation L1 metric.

Reviewer tkcT #1: our reported max pairwise L1 distance between genre routing
distributions (0.60-1.56) is never compared against a chance level. This script
computes the null distribution by:

  1. Loading a trained HR-variant checkpoint.
  2. Running each of the six genre passages token-by-token through the model,
     capturing the router's softmax distribution at every position (not just
     the mean).
  3. Pooling all (position, distribution) rows across all six genres.
  4. Building the OBSERVED per-genre mean distributions and their max pairwise L1.
  5. For each of N=1000 permutations: randomly re-assign the pool of position
     rows to six "pseudo-genres" of the same sizes as the real genres, compute
     the max pairwise L1 for THAT partition, and record it.
  6. Report the null 50th/95th/99th percentiles and the empirical p-value of
     the observed L1 under the null.

If the observed L1 sits far above the null 99th percentile, the specialisation
is not a measurement artifact of the L1 metric or of finite-sample noise.

Runs entirely on CPU or a single GPU (inference only). ~5 min per checkpoint
on the 4060, plus a few seconds per 1000 permutations.

Usage:
    py scripts/permutation_null_l1.py \
        --ckpt runs/tiny_hr/ckpt_final.pt \
        --use_hormones \
        --n_permutations 1000 \
        --layer 11
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


GENRES = {
    "news":    "BREAKING: Officials confirmed Tuesday that the regulatory commission has approved the merger pending shareholder vote next month. Analysts expect the decision to reshape the sector heading into the next fiscal quarter, with several rival firms already issuing statements about competitive responses.",
    "fiction": "She walked through the garden in the late afternoon light, watching the bees move from flower to flower with a quiet patience she had nearly forgotten she was capable of. The sun was warm on her shoulders and somewhere a wood pigeon called out twice and then went still.",
    "code":    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "dialog":  "Yeah I dunno man, like, I told her three times already and she just keeps doing it. What am I supposed to do at this point? I can't keep having the same conversation. It's exhausting and honestly kind of insulting.",
    "tech":    "The transformer architecture relies on scaled dot-product attention, where query and key projections are computed in parallel across heads. The output of each attention head is concatenated and projected back to the model dimension before passing through a feedforward block.",
    "urgent":  "FIRE IN THE NORTH WING. Evacuate immediately via the east staircase. Do not use elevators. Repeat: do not use elevators. Account for all personnel at the marshalling point in the parking lot. Confirm your team lead has eyes on you before leaving the building.",
}


def max_pairwise_l1(dists_by_genre: np.ndarray) -> float:
    """dists_by_genre shape (n_genres, n_hormones), each row sums to 1."""
    diff = np.abs(dists_by_genre[:, None, :] - dists_by_genre[None, :, :]).sum(-1)
    return float(diff.max())


@torch.no_grad()
def capture_per_position_routing(model, tokenizer, layer_idx: int, device: str):
    """For each genre, return an array of shape (n_positions, n_hormones)
    with the router's softmax output at every token position of that layer.

    Returns:
        pooled_rows: np.ndarray of shape (total_positions, n_hormones)
        genre_sizes: dict genre -> n_positions for that genre
    """
    captured: dict[str, np.ndarray] = {}

    def make_hook(genre_key):
        def fn(mod, inp, out):  # noqa: ARG001
            h = inp[0]  # (1, T, d_model)
            with torch.no_grad():
                logits = mod.router(h)                     # (1, T, n_hormones)
                w = torch.softmax(logits, dim=-1)
                w = w[0].detach().cpu().numpy()            # (T, n_hormones)
            captured[genre_key] = w
        return fn

    layer = model.layers[layer_idx]
    if not (hasattr(layer, "hormone") and layer.hormone is not None):
        raise ValueError(f"Layer {layer_idx} has no hormone module")

    all_rows = []
    genre_sizes = {}
    for genre, text in GENRES.items():
        captured.clear()
        handle = layer.hormone.register_forward_hook(make_hook(genre))
        try:
            ids = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)
            _ = model(ids)
        finally:
            handle.remove()
        rows = captured[genre]                              # (T, n_hormones)
        all_rows.append(rows)
        genre_sizes[genre] = rows.shape[0]

    pooled = np.concatenate(all_rows, axis=0)               # (total_T, n_hormones)
    return pooled, genre_sizes


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    p.add_argument("--use_hormones", action="store_true", default=True,
                   help="required for HR variants")
    p.add_argument("--layer", type=int, default=11,
                   help="attention layer index (11 = upper, 5 = lower)")
    p.add_argument("--n_permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None,
                   help="optional .json output path")
    args = p.parse_args()

    # Tokenizer
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    class _TikTok:
        def encode(self, text):
            return enc.encode_ordinary(text)

    tok = _TikTok()

    # Model
    cfg = unnatam_tiny()
    cfg.vocab_size = 50257
    cfg.use_hormones = args.use_hormones
    m = Unnatam(cfg).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    missing, _ = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 3): {missing[:3]}")

    # Capture per-position routing
    print(f"[perm] capturing routing at layer {args.layer} for {len(GENRES)} genres…")
    pooled, sizes = capture_per_position_routing(m, tok, args.layer, args.device)
    total_T = pooled.shape[0]
    print(f"[perm] pooled {total_T} positions across genres: {sizes}")

    # Observed statistic
    observed_dists = np.stack([
        pooled[sum(list(sizes.values())[:i]) : sum(list(sizes.values())[:i]) + sz].mean(axis=0)
        for i, sz in enumerate(sizes.values())
    ])
    observed_l1 = max_pairwise_l1(observed_dists)
    print(f"[perm] observed max pairwise L1 = {observed_l1:.4f}")

    # Null via permutation of position labels
    rng = np.random.default_rng(args.seed)
    genre_sizes_arr = np.array(list(sizes.values()))
    n_perm = args.n_permutations
    null_l1s = np.zeros(n_perm, dtype=np.float32)

    for i in range(n_perm):
        perm = rng.permutation(total_T)
        # Partition pool into genre-sized chunks by permuted index
        chunks = []
        cursor = 0
        for sz in genre_sizes_arr:
            idx = perm[cursor : cursor + sz]
            chunks.append(pooled[idx].mean(axis=0))
            cursor += sz
        null_l1s[i] = max_pairwise_l1(np.stack(chunks))

    q50 = float(np.percentile(null_l1s, 50))
    q95 = float(np.percentile(null_l1s, 95))
    q99 = float(np.percentile(null_l1s, 99))
    p_value = float(np.mean(null_l1s >= observed_l1))

    print(f"\n[perm] null distribution over {n_perm} shuffles:")
    print(f"  median (50%):  {q50:.4f}")
    print(f"  95%ile:        {q95:.4f}")
    print(f"  99%ile:        {q99:.4f}")
    print(f"  observed:      {observed_l1:.4f}")
    print(f"  empirical p:   {p_value:.4f}")
    if observed_l1 > q99:
        print(f"  → observed L1 > 99%ile of null: specialisation is not a metric artifact.")
    else:
        print(f"  → observed L1 within null range: specialisation may be measurement noise.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt,
            "layer": args.layer,
            "n_permutations": n_perm,
            "genre_sizes": sizes,
            "observed_l1": observed_l1,
            "null_median": q50,
            "null_95": q95,
            "null_99": q99,
            "empirical_p": p_value,
            "null_l1s": null_l1s.tolist(),
        }, indent=2))
        print(f"[perm] wrote {args.out}")


if __name__ == "__main__":
    main()
