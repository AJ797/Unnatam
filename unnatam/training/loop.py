"""Training loop. bf16 mixed precision, gradient accumulation, gradient
checkpointing, periodic eval, JSONL logging, periodic checkpointing.

Designed to be importable from a CLI script and from unit tests."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unnatam.training.checkpoint import save_checkpoint
from unnatam.training.optim import build_lr_scheduler, build_optimizer


@dataclass
class TrainConfig:
    # Optim
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    total_steps: int = 1000
    warmup_steps: int = 100

    # Batching
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    seq_len: int = 1024

    # Precision / memory
    dtype: str = "bfloat16"  # "bfloat16" or "float32"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool | None = None  # None = auto (8-bit on CUDA, standard on CPU)

    # Eval / logging / ckpt
    eval_interval: int = 0          # 0 disables eval
    eval_iters: int = 20
    log_interval: int = 10
    ckpt_interval: int = 0          # 0 disables periodic ckpt
    ckpt_dir: str | None = None

    # Misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Filled at runtime
    log_path: str | None = None
    extra: dict = field(default_factory=dict)


def _amp_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _infinite(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: Iterable, cfg: TrainConfig) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = cfg.device
    amp = _amp_dtype(cfg.dtype)
    losses = []
    it = iter(val_loader)
    for _ in range(cfg.eval_iters):
        try:
            batch = next(it)
        except StopIteration:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=amp, enabled=(amp != torch.float32)):
            logits = model(ids)
            loss = compute_lm_loss(logits, labels)
        losses.append(loss.item())
    if was_training:
        model.train()
    if not losses:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}
    avg = sum(losses) / len(losses)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20.0))}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    val_loader: DataLoader | None = None,
    start_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    amp = _amp_dtype(cfg.dtype)

    model.to(device)
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = cfg.gradient_checkpointing

    if optimizer is None:
        optimizer = build_optimizer(
            model, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=cfg.betas, use_8bit=cfg.use_8bit_adam,
        )
    if scheduler is None:
        scheduler = build_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio)

    # Fast-forward scheduler if resuming (LambdaLR is cheap to step).
    for _ in range(start_step):
        scheduler.step()

    log_fp = open(cfg.log_path, "a") if cfg.log_path else None

    def _log(record: dict) -> None:
        msg = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(json.dumps(record) + "\n")
            log_fp.flush()

    model.train()
    train_iter = _infinite(train_loader)
    last_loss = float("nan")
    t0 = time.time()

    for step in range(start_step, cfg.total_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = next(train_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.split(":")[0], dtype=amp, enabled=(amp != torch.float32)
            ):
                logits = model(ids)
                loss = compute_lm_loss(logits, labels) / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        last_loss = accum_loss

        if cfg.log_interval and (step % cfg.log_interval == 0 or step == cfg.total_steps - 1):
            dt = time.time() - t0
            tokens = (step + 1 - start_step) * cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len
            _log({
                "step": step,
                "loss": last_loss,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": dt,
                "tok_per_s": tokens / max(dt, 1e-6),
            })

        if cfg.eval_interval and val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg)
            _log({"step": step, **metrics})

        if cfg.ckpt_interval and cfg.ckpt_dir and step > 0 and step % cfg.ckpt_interval == 0:
            save_checkpoint(
                Path(cfg.ckpt_dir) / f"ckpt_step{step}.pt", model, optimizer, scheduler, step
            )

    if cfg.ckpt_dir:
        save_checkpoint(
            Path(cfg.ckpt_dir) / "ckpt_final.pt", model, optimizer, scheduler, cfg.total_steps
        )

    if log_fp is not None:
        log_fp.close()

    return {"final_loss": last_loss}
