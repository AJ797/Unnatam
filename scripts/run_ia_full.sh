#!/usr/bin/env bash
# MVP IA training run for the NeurIPS submission.
#
# Goal: minimal viable evidence that intracellular attention (a) trains stably,
# (b) opens its gate, and (c) tracks or beats the Base loss curve at the same
# Stage-2 step. Cost-budgeted at ~$30-90 of GPU time.
#
# Strategy:
#   - Reuse the existing Stage-1 standard checkpoint (1B tokens, no IA).
#     The IA gate is zero-init, so attaching IA to the Stage-1 std checkpoint
#     is mathematically a no-op at initialisation; the gate opens during Stage 2.
#     This avoids ~3 days of redundant Stage-1 training.
#   - Train ONLY the IA variant (no `Both`).
#   - 50M Stage-2 tokens — enough for 2-3 eval intervals to plot a trend.
#
# Comparison: existing tiny_base/train.jsonl already logs val loss every
# eval_interval steps from the same Stage-1 ckpt. Plot IA vs Base val loss at
# matching step counts (no separate Base re-run needed).
#
# Wall-clock estimate on 8× RTX PRO 6000S 96GB:
#   - 50M tokens at ~4000 tok/s ≈ 3.5h ≈ $32 at $9/hr
#   - if throughput is closer to 1500 tok/s, ≈ 9h ≈ $80
#
# Run a throughput smoke test first (--tokens 5M) before committing to 50M.
#
# Usage:
#   bash scripts/run_ia_full.sh                       # 50M IA training
#   STAGE2_TOKENS=5M bash scripts/run_ia_full.sh      # smoke test only
#   STAGE2_TOKENS=100M bash scripts/run_ia_full.sh    # bumped budget

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
GPUS="${GPUS:-8}"
STAGE2_TOKENS="${STAGE2_TOKENS:-50M}"
MICRO_BATCH="${MICRO_BATCH:-16}"   # 96GB VRAM lets us use much bigger batches
GRAD_ACCUM="${GRAD_ACCUM:-1}"
IA_STRIDE="${IA_STRIDE:-32}"
WARMUP="${WARMUP:-200}"            # short warmup for the small Stage 2
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
# ──────────────────────────────────────────────────────────────────────────────

# Reused artefact from the original ablation run
STAGE1_STD="${RUNS_DIR}/tiny_stage1/ckpt_milestone.pt"

# Sanity check
if [ ! -f "${STAGE1_STD}" ]; then
    echo "ERROR: missing Stage-1 checkpoint at ${STAGE1_STD}"
    echo "  This script reuses the Stage-1 std checkpoint from the original ablation run."
    echo "  Confirm the workspace was copied correctly from the previous rig."
    exit 1
fi

LAUNCH="torchrun --nproc_per_node=${GPUS} --master_port=29500"
COMMON="--size tiny --vocab_size 50257 \
    --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val \
    --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM} \
    --warmup_steps ${WARMUP} --ckpt_interval ${CKPT_INTERVAL} \
    --log_interval ${LOG_INTERVAL} --eval_interval ${EVAL_INTERVAL} \
    --no_grad_ckpt"

mkdir -p "${RUNS_DIR}"
log() { echo ""; echo "═══════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════"; echo ""; }

# Effective batch sanity
EFF=$((MICRO_BATCH * GRAD_ACCUM * GPUS * 1024))
echo "Effective batch per opt step: ${EFF} tokens (mb=${MICRO_BATCH} ga=${GRAD_ACCUM} gpus=${GPUS} seq=1024)"
echo "Stage 2 budget: ${STAGE2_TOKENS} (Stage 1 std reused: ${STAGE1_STD})"

# =============================================================================
# IA: continue from Stage-1 std with IA enabled (gate zero-init)
# =============================================================================
IA_DIR="${RUNS_DIR}/tiny_ia"
if [ ! -f "${IA_DIR}/ckpt_final.pt" ]; then
    log "IA Stage-2 (continuation from Stage-1 std, ${STAGE2_TOKENS} tokens, IA enabled)"
    ${LAUNCH} scripts/train.py ${COMMON} \
        --intra_attn --ia_stride ${IA_STRIDE} \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_STD}" \
        --out "${IA_DIR}"
else
    log "IA Stage-2: ckpt_final.pt found, skipping"
fi

# =============================================================================
# Done
# =============================================================================
log "IA TRAINING DONE"
if [ -f "${IA_DIR}/ckpt_final.pt" ]; then
    v=$(grep -oP '"val_loss":\s*\K[\d.]+' ${IA_DIR}/train.jsonl 2>/dev/null | tail -1)
    echo "  ✓ tiny_ia  final val_loss=${v}"
else
    echo "  ✗ tiny_ia  MISSING"
fi

echo ""
echo "Next steps:"
echo "  1. Inspect IA gates:        bash scripts/inspect_ia_gates.sh"
echo "  2. Compare val loss curves: bash scripts/compare_ia_vs_base.sh"
echo "  3. Run zero-shot evals:     CUDA_VISIBLE_DEVICES=0 python3 scripts/run_evals.py \\"
echo "                                 --ckpt ${IA_DIR}/ckpt_final.pt --size tiny \\"
echo "                                 --vocab_size 50257 --use_intra_attn \\"
echo "                                 --out ${IA_DIR}/evals.json"
