"""CLI entry for pretraining Unnatam.

Sizes
-----
    tiny   ~125M   fast ablations, 1×4090 friendly
    small  ~350M   paper sweet spot, 4×4090 recommended
    medium ~770M   strong research model, 4×4090 comfortable
    large  ~1.4B   flex model, 4×4090 + patience

Multi-GPU (DDP)
---------------
    torchrun --nproc_per_node=4 scripts/train.py --size tiny --tokens 4B ...
    # LOCAL_RANK / WORLD_SIZE / RANK set automatically by torchrun

Variant flags (ablation matrix)
--------------------------------
    # Base   (no hormones, no IA)
    torchrun ... train.py --size tiny --tokens 4B --out runs/base

    # IA     (intracellular attention only)
    torchrun ... train.py --size tiny --tokens 4B --intra_attn --out runs/ia

    # HR     (hormone routing, extracted vectors; staged)
    #   Stage 1:
    torchrun ... train.py --size tiny --tokens 4B --out runs/stage1 --milestone_tokens 1B
    #   Extraction (single GPU, fast):
    python scripts/extract_hormones.py --ckpt runs/stage1/ckpt_milestone.pt --size tiny --out hormones.npy
    #   Stage 2:
    torchrun ... train.py --size tiny --tokens 3B --hormones --hormone_path hormones.npy \\
        --resume runs/stage1/ckpt_milestone.pt --out runs/hr

    # HR-rand  (random unit-norm vectors)
    torchrun ... train.py --size tiny --tokens 3B --hormones --rand_hormones \\
        --resume runs/stage1/ckpt_milestone.pt --out runs/hr_rand

    # HR-noext (extract at step 0, train full 4B)
    python scripts/extract_hormones.py --init --size tiny --out hormones_noext.npy
    torchrun ... train.py --size tiny --tokens 4B --hormones --hormone_path hormones_noext.npy \\
        --out runs/hr_noext

    # HR-fixedgate (gate=1, frozen)
    torchrun ... train.py --size tiny --tokens 3B --hormones --hormone_path hormones.npy \\
        --fixed_gate --resume runs/stage1/ckpt_milestone.pt --out runs/hr_fixedgate

    # Both (IA + HR)
    #   Stage 1 (IA scan, slow):
    torchrun ... train.py --size tiny --tokens 4B --intra_attn --out runs/ia_stage1 --milestone_tokens 1B
    #   Extract:
    python scripts/extract_hormones.py --ckpt runs/ia_stage1/ckpt_milestone.pt --size tiny \\
        --intra_attn --out hormones_ia.npy
    #   Stage 2:
    torchrun ... train.py --size tiny --tokens 3B --intra_attn --hormones \\
        --hormone_path hormones_ia.npy --resume runs/ia_stage1/ckpt_milestone.pt --out runs/both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from unnatam.config import UnnatamConfig, unnatam_tiny, unnatam_small, unnatam_medium, unnatam_large
from unnatam.model import Unnatam
from unnatam.training.checkpoint import load_checkpoint
from unnatam.training.data import BinaryShardDataset, SyntheticTokenDataset, build_dataloader
from unnatam.training.loop import TrainConfig, train


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp() -> tuple[int, int, int]:
    """Initialise NCCL process group if launched via torchrun.

    Returns (rank, local_rank, world_size).  On single-process runs all are 0/1.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def _teardown_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_tokens(s: str) -> int:
    """Accept '4B', '1_000_000_000', '1e9', '4000000000', etc."""
    s = s.strip().upper().replace("_", "")
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


# ---------------------------------------------------------------------------
# Config / model builders
# ---------------------------------------------------------------------------

def _build_model_config(args: argparse.Namespace) -> UnnatamConfig:
    cfg: UnnatamConfig = {
        "tiny":   unnatam_tiny,
        "small":  unnatam_small,
        "medium": unnatam_medium,
        "large":  unnatam_large,
    }[args.size]()

    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    # Intracellular attention
    cfg.use_intra_attn = args.intra_attn
    cfg.ia_stride = args.ia_stride

    # Hormone routing
    cfg.use_hormones = args.hormones
    if args.hormones:
        if args.rand_hormones:
            cfg.hormone_random = True
        elif args.hormone_path:
            cfg.hormone_vector_path = args.hormone_path
        # else: random small-magnitude init (gate=0 keeps it inert until extraction)
        cfg.hormone_fixed_gate = args.fixed_gate
        cfg.hormone_router_init_gate = float(args.init_gate)
        cfg.hormone_alpha = float(args.alpha)

    return cfg


def _build_dataset(args: argparse.Namespace, cfg: UnnatamConfig, rank: int, world_size: int):
    if args.data == "synthetic":
        return SyntheticTokenDataset(
            vocab_size=cfg.vocab_size, seq_len=args.seq_len,
            seed=args.seed + rank,  # each rank sees different data
        )
    data_dir = Path(args.data)
    shards = sorted(data_dir.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"no .bin shards under {data_dir}")
    dtype = np.uint16 if cfg.vocab_size <= 65536 else np.uint32
    return BinaryShardDataset(
        shards, seq_len=args.seq_len, dtype=dtype,
        seed=args.seed + rank * 997,  # coprime offset keeps rank seeds far apart
    )


def _build_val_dataset(args: argparse.Namespace, cfg: UnnatamConfig):
    if not args.val_data:
        return None
    val_dir = Path(args.val_data)
    shards = sorted(val_dir.glob("*.bin"))
    if not shards:
        return None
    dtype = np.uint16 if cfg.vocab_size <= 65536 else np.uint32
    return BinaryShardDataset(shards, seq_len=args.seq_len, dtype=dtype, seed=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model
    p.add_argument("--size", choices=["tiny", "small", "medium", "large"], default="small")
    p.add_argument("--vocab_size", type=int, default=None,
                   help="override vocab_size (default: 50257 from config)")

    # Data
    p.add_argument("--data", default="synthetic",
                   help="'synthetic' or path to directory of tokenized .bin shards")
    p.add_argument("--val_data", default=None,
                   help="path to directory of validation .bin shards (optional)")
    p.add_argument("--seq_len", type=int, default=1024)

    # Training budget — specify EITHER --tokens or --total_steps
    p.add_argument("--tokens", type=str, default=None,
                   help="total training tokens, e.g. '4B' or '1_000_000_000' (overrides --total_steps)")
    p.add_argument("--total_steps", type=int, default=50000,
                   help="total optimizer steps (ignored when --tokens is given)")

    # Milestone checkpoint (for staged training)
    p.add_argument("--milestone_tokens", type=str, default=None,
                   help="save ckpt_milestone.pt when this many tokens have been seen, e.g. '1B'")

    # Batch / optim
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")

    # Architecture flags
    p.add_argument("--intra_attn", action="store_true",
                   help="enable intracellular attention in SSM blocks (forces ref scan)")
    p.add_argument("--ia_stride", type=int, default=32,
                   help="apply intracellular attention every K scan steps (1=every step, default 32)")

    # Hormone variant flags
    p.add_argument("--hormones", action="store_true",
                   help="enable hormone routing (required for HR / Both / HR-* variants)")
    p.add_argument("--hormone_path", type=str, default=None,
                   help="path to pre-extracted .npy hormone bank (n_hormones, d_model)")
    p.add_argument("--rand_hormones", action="store_true",
                   help="use random unit-norm vectors instead of extracted (HR-rand ablation)")
    p.add_argument("--fixed_gate", action="store_true",
                   help="freeze hormone router gate at 1.0 — non-trainable (HR-fixedgate ablation)")
    p.add_argument("--init_gate", type=float, default=0.0,
                   help="initial value of hormone-router output gate (0.0 = no-op at init; "
                        ">0 = warm start, the model feels the signal from step 0 — HR-forced)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="constant multiplier on the hormone injection (1.0 = baseline; "
                        ">1.0 amplifies — used together with --init_gate for HR-forced)")

    # Dataloader
    p.add_argument("--num_workers", type=int, default=4,
                   help="dataloader workers per rank (auto-clamped to cpu_count/world_size)")

    # Memory
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--no_8bit_adam", action="store_true",
                   help="force standard AdamW (for debugging on CPU)")

    # Logging / checkpointing
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--ckpt_interval", type=int, default=5000)
    p.add_argument("--out", "--output_dir", dest="output_dir", default="runs/default")
    p.add_argument("--run_name", type=str, default=None,
                   help="optional name prefix added to log lines")

    # Resumption
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume from")

    # Misc
    p.add_argument("--device", type=str, default=None,
                   help="device override (default: cuda:LOCAL_RANK or cpu)")
    p.add_argument("--seed", type=int, default=1337)

    args = p.parse_args()

    # -----------------------------------------------------------------------
    # DDP init
    # -----------------------------------------------------------------------
    rank, local_rank, world_size = _setup_ddp()

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    is_main = (rank == 0)

    # -----------------------------------------------------------------------
    # Config / model
    # -----------------------------------------------------------------------
    cfg = _build_model_config(args)

    # Compute total_steps from token budget if given.
    tokens_per_step = args.micro_batch_size * args.grad_accum_steps * args.seq_len * world_size
    if args.tokens is not None:
        total_tokens = _parse_tokens(args.tokens)
        total_steps = max(1, total_tokens // tokens_per_step)
    else:
        total_steps = args.total_steps
        total_tokens = total_steps * tokens_per_step

    milestone_tokens = _parse_tokens(args.milestone_tokens) if args.milestone_tokens else 0

    if is_main:
        print(f"[unnatam] size={args.size} d_model={cfg.d_model} n_layers={cfg.n_layers} "
              f"n_attn={cfg.n_attn_layers} vocab={cfg.vocab_size}")
        print(f"[unnatam] world_size={world_size} device={device}")
        print(f"[unnatam] tokens_per_step={tokens_per_step:,} total_steps={total_steps:,} "
              f"total_tokens={total_tokens/1e9:.2f}B")
        ia_str = f"intra_attn=True (stride={args.ia_stride})" if args.intra_attn else "intra_attn=False"
        print(f"[unnatam] hormones={args.hormones} {ia_str} "
              f"rand_hormones={args.rand_hormones} fixed_gate={args.fixed_gate}")

    model = Unnatam(cfg)

    # Apply HR-fixedgate BEFORE DDP wrapping so gate is already frozen when
    # DDP wraps the parameters.
    if args.hormones and args.fixed_gate:
        model.freeze_hormone_gates(gate_value=1.0)

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = model.num_parameters()
        print(f"[unnatam] params total={total_p/1e6:.1f}M trainable={trainable/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_dataset = _build_dataset(args, cfg, rank, world_size)
    val_dataset = _build_val_dataset(args, cfg)

    # Per-rank num_workers: aim for ~4-8 workers per rank but never more
    # than (cpu_count // world_size) so we don't over-subscribe.
    cpu_count = os.cpu_count() or 8
    per_rank_workers = max(1, min(args.num_workers, cpu_count // max(1, world_size)))
    if is_main:
        print(f"[unnatam] dataloader: num_workers={per_rank_workers} per rank "
              f"(cpu_count={cpu_count}, world_size={world_size})")

    train_loader = build_dataloader(
        train_dataset, batch_size=args.micro_batch_size,
        num_workers=per_rank_workers, pin_memory=(device != "cpu"),
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = (
        build_dataloader(val_dataset, batch_size=args.micro_batch_size,
                         num_workers=1, pin_memory=(device != "cpu"),
                         persistent_workers=True, prefetch_factor=2)
        if val_dataset is not None else None
    )

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    model.to(device)
    start_step = 0
    if args.resume:
        # Load before DDP wrapping (DDP expects unwrapped state_dict).
        loaded_step = load_checkpoint(args.resume, model, map_location=device)
        # Heuristic: if the loaded ckpt is at or past our planned total_steps,
        # this is a cross-stage transition (e.g. Stage-1 → Stage-2). Reset the
        # step counter so the new training budget is interpreted as fresh
        # additional steps with a fresh LR schedule. Otherwise (mid-run resume
        # after a crash) preserve the step count so we don't redo work.
        if loaded_step >= total_steps:
            start_step = 0
            if is_main:
                print(f"[unnatam] resumed from {args.resume} at step {loaded_step} "
                      f"(>= total_steps {total_steps}); treating as cross-stage "
                      f"transition: resetting step counter & LR schedule")
        else:
            start_step = loaded_step
            if is_main:
                print(f"[unnatam] resumed from {args.resume} at step {start_step} "
                      f"(continuing toward total_steps={total_steps})")

    # -----------------------------------------------------------------------
    # DDP wrap
    # -----------------------------------------------------------------------
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # -----------------------------------------------------------------------
    # Output dir
    # -----------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        dtype=args.dtype,
        gradient_checkpointing=(not args.no_grad_ckpt),
        use_8bit_adam=(False if args.no_8bit_adam else None),
        milestone_tokens=milestone_tokens,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        ckpt_dir=str(out_dir) if is_main else None,
        seed=args.seed,
        device=device,
        log_path=str(out_dir / "train.jsonl") if is_main else None,
    )

    train(model, train_loader, train_cfg,
          val_loader=val_loader,
          start_step=start_step)

    _teardown_ddp(world_size)


if __name__ == "__main__":
    main()
