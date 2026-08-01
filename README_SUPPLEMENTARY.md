# Supplementary Code

This is the anonymised code release accompanying the paper *"When Structure Doesn't Matter: Loss-Invariant Routing Specialisation in Hybrid Language Models."*

## Contents

```
unnatam/                     # the model package
  config.py                  # backbone + ablation config dataclasses
  model/                     # hybrid Mamba/MQA backbone
    ssm.py                   # selective scan (with optional intracellular attention)
    attention.py             # multi-query attention with RoPE
    block.py                 # SSMLayer, AttnLayer (with hormone injection)
    hormone.py               # HormoneRouter module
    unnatam.py               # top-level model + hormone bank loading
    mlp.py, norm.py          # SwiGLU, RMSNorm
  hormones/
    definitions.py           # the 7 × 6 = 42 contrast pairs (full text)
    extract.py               # contrast-pair extraction algorithm
  training/
    loop.py                  # DDP training loop, eval, checkpointing
    data.py                  # binary-shard dataloader
    optim.py                 # AdamW 8-bit + cosine LR schedule
    checkpoint.py            # checkpoint save/load with cross-stage handling
  benchmarks/
    eval.py                  # zero-shot loglikelihood evaluator (HellaSwag/LAMBADA/ARC/PIQA)

scripts/
  train.py                   # DDP training entry point
  prepare_data.py            # FineWeb-Edu tokenisation → binary shards
  extract_hormones.py        # one-shot hormone extraction
  run_evals.py               # zero-shot eval CLI
  make_paper_plots.py        # regenerate all paper figures from train.jsonl + ckpts

tests/                       # smoke and unit tests
configs/                     # YAML overrides used in the paper
pyproject.toml               # package metadata
requirements.txt             # pip dependencies
```

## Setup

```bash
pip install -e .
pip install -r requirements.txt

# Optional fused kernels (recommended for non-IA training, ~25× speedup):
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir causal-conv1d
MAMBA_FORCE_BUILD=TRUE        pip install --no-build-isolation --no-cache-dir mamba-ssm

# Verify
python -c "from unnatam.model.ssm import has_fast_scan; print('fast_scan:', has_fast_scan())"
```

## Reproducing the paper

### 1. Tokenise FineWeb-Edu

```bash
python scripts/prepare_data.py --out data/fineweb \
    --num_tokens 10_000_000_000 --val_tokens 50_000_000 --n_workers 8
```

### 2. Train Stage 1 (1B tokens, no hormones)

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --tokens 1B --milestone_tokens 1B \
    --micro_batch_size 8 --grad_accum_steps 2 \
    --warmup_steps 2000 --ckpt_interval 5000 \
    --out runs/tiny_stage1
```

### 3. Extract hormones from the milestone checkpoint

```bash
python scripts/extract_hormones.py \
    --ckpt runs/tiny_stage1/ckpt_milestone.pt \
    --size tiny --vocab_size 50257 \
    --out runs/tiny_hormones.npy
```

### 4. Train Stage-2 variants

For each variant, resume from the Stage-1 checkpoint:

```bash
# Base (continuation, no hormones)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --tokens 1B --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_base

# HR (extracted hormones, learnable gate, init=0)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --hormones --hormone_path runs/tiny_hormones.npy \
    --tokens 1B --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_hr

# HR-rand (random unit vectors)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --hormones --rand_hormones \
    --tokens 1B --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_hr_rand

# HR-fixedgate (gate frozen at 1.0)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --hormones --hormone_path runs/tiny_hormones.npy --fixed_gate \
    --tokens 1B --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_hr_fixedgate

# HR-noext (random-init extracted hormones, FROM SCRATCH)
python scripts/extract_hormones.py --init --size tiny --vocab_size 50257 \
    --out runs/tiny_hormones_noext.npy
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --hormones --hormone_path runs/tiny_hormones_noext.npy \
    --tokens 2B --out runs/tiny_hr_noext

# HR-forced (init_gate=0.2, alpha=3, continuation)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --hormones --hormone_path runs/tiny_hormones.npy \
    --init_gate 0.2 --alpha 3.0 \
    --tokens 200M --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_hr_forced

# IA (intracellular attention, reference scan, reduced budget)
torchrun --nproc_per_node=8 scripts/train.py --size tiny --vocab_size 50257 \
    --data data/fineweb/train --val_data data/fineweb/val \
    --intra_attn --ia_stride 32 \
    --micro_batch_size 2 --grad_accum_steps 4 \
    --warmup_steps 50 --no_grad_ckpt \
    --tokens 12M --resume runs/tiny_stage1/ckpt_milestone.pt \
    --out runs/tiny_ia
```

### 5. Run zero-shot evals

```bash
for variant in tiny_base tiny_hr tiny_hr_rand tiny_hr_fixedgate tiny_hr_noext tiny_hr_forced; do
    python scripts/run_evals.py \
        --ckpt runs/${variant}/ckpt_final.pt \
        --size tiny --vocab_size 50257 --use_hormones \
        --out runs/${variant}/evals.json
done
python scripts/run_evals.py --ckpt runs/tiny_ia/ckpt_final.pt \
    --size tiny --vocab_size 50257 --use_intra_attn \
    --out runs/tiny_ia/evals.json
```

### 6. Regenerate paper figures

```bash
python scripts/make_paper_plots.py --runs runs/ --out figs/
```

## Hardware

We trained on rented 8× RTX 4090/5090 (24 GB) instances. Total wall-clock:
- Stage 1: ~12h
- Each Stage-2 hormone variant: ~12h
- HR-forced (200M): ~30 min
- IA (12M, reference scan): ~9h
- Evals (single GPU): ~1h per variant

Total compute budget for the reported results was approximately 400 GPU-hours.

## Smoke tests

```bash
pytest tests/        # unit tests, no GPU required
python scripts/train.py --size tiny --data synthetic --total_steps 10 --device cuda:0  # 30s GPU smoke
```
