"""Hormone vector extraction.

Algorithm (per hormone):
  1. For each contrast pair (positive_text, negative_text):
       - Forward each text through the model
       - Capture residual stream output at every AttnLayer (the injection points)
       - Mean across attention layers, then mean across token positions
         → one (d_model,) vector per text
       - Compute (pos_vec - neg_vec)  → the per-pair direction
  2. Mean the pair-directions  → the hormone direction
  3. L2-normalize to unit length

We extract at exactly the layers where injection happens, so the directions
live in the same residual subspace they'll be added back into. Magnitudes are
normalized to 1.0 so the per-hormone learned magnitudes (in HormoneRouter)
have a clean scale to train against.

Tokenizer protocol: any object with `.encode(str) -> list[int]` works.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unnatam.hormones.definitions import HORMONE_CONTRASTS, HORMONE_NAMES
from unnatam.model.block import AttnLayer


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


@torch.no_grad()
def extract_hormone_vectors(
    model: nn.Module,
    tokenizer: TokenizerLike,
    device: str | torch.device = "cpu",
    max_tokens: int = 256,
    contrasts: dict[str, list] | None = None,
) -> np.ndarray:
    """Run the contrast pairs through `model` and return frozen hormone direction vectors.

    Returns a numpy array of shape (n_hormones, d_model), L2-normalized along the
    last axis. Order matches `HORMONE_NAMES`.
    """
    contrasts = contrasts if contrasts is not None else HORMONE_CONTRASTS
    if not contrasts:
        raise ValueError("contrasts dict is empty")

    was_training = model.training
    model.eval()
    model.to(device)

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output)

    handles = []
    for layer in model.layers:
        if isinstance(layer, AttnLayer):
            handles.append(layer.register_forward_hook(hook))
    if not handles:
        raise ValueError("model has no AttnLayer instances to hook for extraction")

    def encode_one(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text)[:max_tokens]
        if not ids:
            raise ValueError(f"tokenizer returned empty ids for text: {text!r}")
        return torch.tensor([ids], device=device, dtype=torch.long)

    def collect(text: str) -> torch.Tensor:
        captured.clear()
        ids = encode_one(text)
        _ = model(ids)
        # captured: list of (1, T, d_model), one per AttnLayer
        stacked = torch.stack(captured, dim=0)         # (n_attn, 1, T, d_model)
        # Mean over attention layers and over token positions.
        return stacked.mean(dim=(0, 2)).squeeze(0).float().cpu()

    try:
        names = [n for n in HORMONE_NAMES if n in contrasts]
        d_model = next(model.parameters()).shape[-1] if hasattr(model, "embed") else None
        if hasattr(model, "cfg"):
            d_model = model.cfg.d_model
        if d_model is None:
            raise RuntimeError("could not determine model d_model")

        vectors = torch.zeros(len(names), d_model, dtype=torch.float32)
        for h_idx, name in enumerate(names):
            pairs = contrasts[name]
            if not pairs:
                raise ValueError(f"no contrast pairs for hormone {name!r}")
            diffs = []
            for pair in pairs:
                pos_v = collect(pair.positive)
                neg_v = collect(pair.negative)
                diffs.append(pos_v - neg_v)
            mean_diff = torch.stack(diffs).mean(dim=0)
            vectors[h_idx] = F.normalize(mean_diff, dim=0)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return vectors.numpy()


def save_hormone_vectors(path: str, vectors: np.ndarray) -> None:
    if vectors.ndim != 2:
        raise ValueError(f"expected (n_hormones, d_model), got shape {vectors.shape}")
    np.save(path, vectors)
