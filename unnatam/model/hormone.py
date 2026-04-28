from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HormoneRouter(nn.Module):
    """Per-attention-layer hormone router.

    The frozen hormone-vector dictionary is shared across layers (passed in at
    forward time). Each instance owns its own router weights, per-hormone
    learned magnitudes, and an output gate (init 0 → no-op at init, similar to
    LoRA's zero-init trick so the base model isn't disrupted at the start of
    training).
    """

    def __init__(self, d_model: int, n_hormones: int, init_gate: float = 0.0):
        super().__init__()
        self.router = nn.Linear(d_model, n_hormones, bias=False)
        self.magnitudes = nn.Parameter(torch.ones(n_hormones))
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, h: torch.Tensor, hormone_vectors: torch.Tensor) -> torch.Tensor:
        """
        h:                (B, T, d_model)
        hormone_vectors:  (n_hormones, d_model), frozen
        returns:          (B, T, d_model) shift to add to residual
        """
        logits = self.router(h)                                       # (B, T, n_hormones)
        weights = F.softmax(logits, dim=-1)
        scaled_v = hormone_vectors * self.magnitudes.unsqueeze(-1)    # (n_hormones, d_model)
        shift = weights @ scaled_v                                    # (B, T, d_model)
        return self.gate * shift
