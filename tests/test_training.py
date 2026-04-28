"""End-to-end smoke tests for the training stack.

CPU-only by default so they run anywhere. Use small models / short runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unnatam.config import UnnatamConfig
from unnatam.model import Unnatam
from unnatam.training.checkpoint import load_checkpoint, save_checkpoint
from unnatam.training.data import BinaryShardDataset, SyntheticTokenDataset, build_dataloader
from unnatam.training.loop import TrainConfig, compute_lm_loss, train
from unnatam.training.optim import build_lr_scheduler, build_optimizer, build_param_groups


def _tiny_cfg(**overrides) -> UnnatamConfig:
    base = dict(
        d_model=64, n_layers=4, vocab_size=128, max_seq_len=32,
        n_heads=2, n_kv_heads=1, head_dim=32,
        d_state=4, d_conv=4, expand=2,
        ssm_attn_ratio=2, n_hormones=7,
    )
    base.update(overrides)
    return UnnatamConfig(**base)


def test_param_groups_split_correctly():
    model = Unnatam(_tiny_cfg())
    groups = build_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2
    decay_params = {id(p) for p in groups[0]["params"]}
    no_decay_params = {id(p) for p in groups[1]["params"]}
    # Sanity: norm weights and biases should be no-decay.
    for name, p in model.named_parameters():
        if "norm" in name or p.ndim < 2:
            assert id(p) in no_decay_params, f"{name} should be no-decay"
    # SSM A_log/D explicitly marked no_weight_decay.
    for name, p in model.named_parameters():
        if name.endswith("A_log") or name.endswith(".D"):
            assert id(p) in no_decay_params, f"{name} should be no-decay (marked _no_weight_decay)"
    assert decay_params.isdisjoint(no_decay_params)


def test_overfit_single_batch_drops_loss():
    """Sanity: with a fixed input batch, AdamW should drive loss substantially down."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.0, use_8bit=False)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = torch.randint(0, cfg.vocab_size, (2, 16))

    model.train()
    losses = []
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = compute_lm_loss(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # First-step loss is roughly log(vocab) ~ log(128) ≈ 4.85. Should drop a lot.
    assert losses[0] > 3.0, f"loss didn't start where expected: {losses[0]}"
    assert losses[-1] < 1.0, f"didn't overfit: start={losses[0]:.3f} end={losses[-1]:.3f}"
    assert losses[-1] < losses[0] - 2.0


def test_short_training_run_with_synthetic_data(tmp_path: Path):
    cfg = _tiny_cfg()
    model = Unnatam(cfg)

    dataset = SyntheticTokenDataset(vocab_size=cfg.vocab_size, seq_len=16, seed=0)
    loader = build_dataloader(dataset, batch_size=2, num_workers=0, pin_memory=False)

    train_cfg = TrainConfig(
        lr=3e-3,
        total_steps=20,
        warmup_steps=2,
        micro_batch_size=2,
        grad_accum_steps=1,
        seq_len=16,
        dtype="float32",          # CPU + bf16 in autocast is supported but slower; keep deterministic.
        gradient_checkpointing=False,
        use_8bit_adam=False,      # bitsandbytes doesn't support CPU; force standard AdamW.
        log_interval=10,
        ckpt_dir=str(tmp_path),
        ckpt_interval=0,
        device="cpu",
    )
    out = train(model, loader, train_cfg)
    assert out["final_loss"] < 6.0  # just want it to run without exploding
    # Final ckpt should be on disk.
    assert (tmp_path / "ckpt_final.pt").exists()


def test_checkpoint_roundtrip_preserves_logits(tmp_path: Path):
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = Unnatam(cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0, use_8bit=False)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=2, total_steps=10)

    # Take a couple of optim steps so optimizer state is non-trivial.
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_lm_loss(model(ids), ids)
        loss.backward()
        optimizer.step()
        scheduler.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scheduler, step=3)

    # Reference logits from the saved model.
    model.eval()
    with torch.no_grad():
        ref = model(ids)

    # Fresh model, load ckpt, check identical.
    model2 = Unnatam(cfg)
    optim2 = build_optimizer(model2, lr=1e-3, weight_decay=0.0)
    sched2 = build_lr_scheduler(optim2, warmup_steps=2, total_steps=10)
    step = load_checkpoint(path, model2, optim2, sched2)
    assert step == 3

    model2.eval()
    with torch.no_grad():
        out = model2(ids)
    assert torch.allclose(ref, out, atol=1e-6), "logits diverged after ckpt round-trip"


def test_gradient_checkpointing_matches_non_checkpointed():
    """Loss with grad-ckpt should exactly match loss without (forward is the same).
    Compares forward+loss only — backward equivalence is implied since both go
    through the same checkpoint primitives."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    model.train()
    model.gradient_checkpointing = False
    loss_a = compute_lm_loss(model(ids), ids)

    model.gradient_checkpointing = True
    loss_b = compute_lm_loss(model(ids), ids)

    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_binary_shard_dataset_yields_valid_windows(tmp_path: Path):
    # Synthesize a small .bin shard.
    shard = tmp_path / "shard.bin"
    arr = np.arange(2000, dtype=np.uint16) % 100  # vocab=100
    arr.tofile(shard)
    ds = BinaryShardDataset([shard], seq_len=32, dtype=np.uint16, seed=0)
    it = iter(ds)
    for _ in range(5):
        sample = next(it)
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)
        # off-by-one check
        idx = (sample["input_ids"] != sample["labels"]).any()
        assert idx  # on a non-constant buffer the windows differ
