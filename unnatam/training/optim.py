"""Optimizer + LR scheduler factories.

Param groups follow the standard recipe: weight decay applied only to 2D+
tensors (Linear weights, embedding matrix), excluded from 1D tensors (biases,
norms) and any parameter explicitly marked `_no_weight_decay` (Mamba's A_log
and D, by convention).

8-bit Adam (bitsandbytes) is used automatically when:
  - bitsandbytes is installed
  - the model is on CUDA
  - use_8bit is not explicitly set to False

This is the only way to fit 770M+ param training into 6 GB VRAM. Standard
AdamW needs ~12 bytes/param for optimizer state (fp32 m + v); bnb.AdamW8bit
needs ~2 bytes/param (quantized m + v). On a 770M model that's the difference
between ~9 GB and ~1.5 GB for optimizer state alone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb

    _HAS_BNB = True
except ImportError:
    bnb = None  # type: ignore[assignment]
    _HAS_BNB = False


def has_8bit_adam() -> bool:
    return _HAS_BNB


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "_no_weight_decay", False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    use_8bit: bool | None = None,  # None = auto-detect
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with automatic 8-bit selection.

    When bitsandbytes is installed and the model is on CUDA, defaults to
    AdamW8bit to conserve VRAM. Pass use_8bit=False to force standard AdamW
    (useful for CPU testing — bnb doesn't support CPU).
    """
    groups = build_param_groups(model, weight_decay)
    on_cuda = torch.cuda.is_available()

    # Auto-detect: 8-bit if bnb available + CUDA, unless explicitly overridden.
    if use_8bit is None:
        use_8bit = _HAS_BNB and on_cuda

    if use_8bit:
        if not _HAS_BNB:
            raise RuntimeError(
                "use_8bit=True but bitsandbytes is not installed. "
                "Run: pip install bitsandbytes"
            )
        optimizer = bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
        # Stabilize embedding table training with bnb — keeps the embed matrix
        # in 32-bit for the optimizer update step.
        if hasattr(model, "embed"):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                model.embed, "weight", {"optim_bits": 32}
            )
    else:
        optimizer = torch.optim.AdamW(
            groups, lr=lr, betas=betas, eps=eps,
            fused=(on_cuda and not use_8bit),
        )
    return optimizer


def cosine_warmup_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """LR multiplier ∈ [min_lr_ratio, 1] given the current step."""
    warmup_steps = max(warmup_steps, 1)
    decay_steps = max(total_steps - warmup_steps, 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_warmup_lambda(warmup_steps, total_steps, min_lr_ratio)
    )
