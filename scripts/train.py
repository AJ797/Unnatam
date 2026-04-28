"""CLI entry for pretraining Unnatam.

Examples
--------
    # quick smoke run on synthetic data, no GPU needed
    python scripts/train.py --size small --data synthetic --total_steps 50 --device cpu

    # real run on pre-tokenized FineWeb shards
    python scripts/train.py --size small --data_dir data/fineweb --total_steps 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from unnatam.config import UnnatamConfig, unnatam_medium, unnatam_small
from unnatam.model import Unnatam
from unnatam.training.checkpoint import load_checkpoint
from unnatam.training.data import BinaryShardDataset, SyntheticTokenDataset, build_dataloader
from unnatam.training.loop import TrainConfig, train


def _build_config(size: str, vocab_size: int | None, seq_len: int) -> UnnatamConfig:
    cfg = {"small": unnatam_small, "medium": unnatam_medium}[size]()
    if vocab_size is not None:
        cfg.vocab_size = vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
    return cfg


def _build_dataset(args: argparse.Namespace, cfg: UnnatamConfig):
    if args.data == "synthetic":
        return SyntheticTokenDataset(vocab_size=cfg.vocab_size, seq_len=args.seq_len, seed=args.seed)
    shards = sorted(Path(args.data_dir).glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"no .bin shards under {args.data_dir}")
    dtype = np.uint16 if cfg.vocab_size <= 65536 else np.uint32
    return BinaryShardDataset(shards, seq_len=args.seq_len, dtype=dtype, seed=args.seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=["small", "medium"], default="small")
    p.add_argument("--data", choices=["synthetic", "shards"], default="shards")
    p.add_argument("--data_dir", type=str, default=None, help="directory of .bin shards (when --data=shards)")
    p.add_argument("--vocab_size", type=int, default=None, help="override config vocab_size to match tokenizer")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--no_8bit_adam", action="store_true", help="force standard AdamW (for debugging)")
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--ckpt_interval", type=int, default=2000)
    p.add_argument("--output_dir", type=str, default="runs/default")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.data == "shards" and not args.data_dir:
        p.error("--data_dir is required when --data=shards")

    cfg = _build_config(args.size, args.vocab_size, args.seq_len)
    print(f"[unnatam] config: d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"attn_layers={cfg.n_attn_layers} vocab_size={cfg.vocab_size}")

    model = Unnatam(cfg)
    print(f"[unnatam] params: {model.num_parameters() / 1e6:.1f}M")

    dataset = _build_dataset(args, cfg)
    loader = build_dataloader(
        dataset, batch_size=args.micro_batch_size, num_workers=0, pin_memory=(args.device != "cpu")
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        dtype=args.dtype,
        gradient_checkpointing=(not args.no_grad_ckpt),
        use_8bit_adam=(False if args.no_8bit_adam else None),
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        ckpt_dir=str(out_dir),
        seed=args.seed,
        device=args.device,
        log_path=str(out_dir / "train.jsonl"),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, map_location=args.device)
        print(f"[unnatam] resumed from {args.resume} at step {start_step}")

    train(model, loader, train_cfg, start_step=start_step)


if __name__ == "__main__":
    main()
