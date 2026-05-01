"""Checkpoint save/load. Stores everything needed for bit-identical resumption."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        # Dataclasses get unwrapped so the payload is plain JSON-style on disk.
        payload["extra"] = {k: (asdict(v) if is_dataclass(v) else v) for k, v in extra.items()}
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> int:
    """Load weights (and optionally optimiser/scheduler state) from a checkpoint.

    ``strict=False`` by default so that we can resume Stage-1 (no-hormone) ckpts
    into Stage-2 models that have additional ``HormoneRouter`` modules — those
    new params keep their init values (zero-init gate keeps them inert).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if missing:
        # Filter to "expected" missing keys (hormone router params when adding HR mid-train)
        expected_prefixes = ("hormone.",)
        unexpected_missing = [k for k in missing if not any(p in k for p in expected_prefixes)]
        if unexpected_missing:
            print(f"[load_checkpoint] WARNING: unexpected missing keys: {unexpected_missing[:5]}"
                  f"{'...' if len(unexpected_missing) > 5 else ''}", flush=True)
        if any(p in k for p in expected_prefixes for k in missing):
            print(f"[load_checkpoint] note: hormone-router params not in ckpt — using init values", flush=True)
    if unexpected:
        print(f"[load_checkpoint] WARNING: unexpected keys in ckpt: {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}", flush=True)
    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[load_checkpoint] could not restore optimizer state ({e}); using fresh optimizer", flush=True)
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))
