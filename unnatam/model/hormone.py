from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HormoneRouter(nn.Module):
    """Per-attention-layer hormone router.

    The frozen hormone-vector dictionary is shared across layers (passed in at
    forward time). Each instance owns its own router weights, per-hormone
    learned magnitudes, and an output gate. The gate is initialised to
    ``init_gate`` (0 → no-op at init, similar to LoRA's zero-init trick;
    >0 → "warm" start that lets the model feel the signal from step 0).

    ``alpha`` is a constant multiplier applied to the gated shift, allowing
    forced amplification of the injection without changing the trainable
    gate's natural range. Useful for the HR-forced ablation where we want
    the model to receive a stronger perturbation throughout training.
    """

    def __init__(self, d_model: int, n_hormones: int,
                 init_gate: float = 0.0, alpha: float = 1.0):
        super().__init__()
        self.router = nn.Linear(d_model, n_hormones, bias=False)
        self.magnitudes = nn.Parameter(torch.ones(n_hormones))
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.alpha = float(alpha)
        # Most recent ||injection|| / ||residual|| ratio per forward pass.
        # Populated when log_signal=True is passed at forward time. Floats so
        # they can be picked up by training-loop loggers without any new state.
        self.last_signal_ratio: float = 0.0

    def forward(self, h: torch.Tensor, hormone_vectors: torch.Tensor,
                log_signal: bool = False) -> torch.Tensor:
        """
        h:                (B, T, d_model)
        hormone_vectors:  (n_hormones, d_model), frozen
        returns:          (B, T, d_model) shift to add to residual
        """
        logits = self.router(h)                                       # (B, T, n_hormones)
        weights = F.softmax(logits, dim=-1)
        scaled_v = hormone_vectors * self.magnitudes.unsqueeze(-1)    # (n_hormones, d_model)
        shift = weights @ scaled_v                                    # (B, T, d_model)
        out = self.alpha * self.gate * shift
        if log_signal:
            with torch.no_grad():
                inj_norm = out.float().norm(dim=-1).mean()            # (B, T) → scalar
                res_norm = h.float().norm(dim=-1).mean()
                self.last_signal_ratio = float(
                    (inj_norm / (res_norm + 1e-8)).item()
                )
        return out
