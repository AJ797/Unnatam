#!/usr/bin/env bash
# Tight IA training plan to fit the NeurIPS deadline.
#
# Strategy:
#   - Reuse the existing Stage-1 standard checkpoint (1B tokens, no IA).
#     The IA gate is zero-init, so attaching IA to the existing checkpoint is
#     mathematically a no-op at initialisation; the gate opens during Stage 2.
#     This saves ~3 days of redundant Stage-1 training.
#   - Reuse the existing extracted hormones (tiny_hormones.npy) for `Both`.
#   - Stage 2 IA and Stage 2 Both each train for 500M tokens (vs 1B for HR
#     variants). This is a known asymmetry — see paper footnote.
#
# Token budgets:
#   - Base / HR / HR-rand / HR-noext / HR-fixedgate: 2B (1B Stage 1 + 1B Stage 2)
#   - IA / Both: 1.5B (1B Stage 1 std + 500M Stage 2 with IA)
#
# Wall-clock estimate on 8× RTX PRO 6000S 96GB:
#   - Stage 2 IA   (500M): ~1.5 days (8 GPUs, mb=16, ia_stride=32)
#   - Stage 2 Both (500M): ~1.5 days (8 GPUs, mb=16, ia_stride=32)
#   - Total: ~3 days sequential
#
# Usage:
#   bash scripts/run_ia_full.sh
#   STAGE2_TOKENS=500M MICRO_BATCH=16 IA_STRIDE=32 bash scripts/run_ia_full.sh

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
GPUS="${GPUS:-8}"
STAGE2_TOKENS="${STAGE2_TOKENS:-500M}"
MICRO_BATCH="${MICRO_BATCH:-16}"   # 96GB VRAM lets us use much bigger batches
GRAD_ACCUM="${GRAD_ACCUM:-1}"
IA_STRIDE="${IA_STRIDE:-32}"
WARMUP="${WARMUP:-1000}"           # shorter warmup for the shorter Stage 2
CKPT_INTERVAL="${CKPT_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
# ──────────────────────────────────────────────────────────────────────────────

# Reused artefacts from the original ablation run
STAGE1_STD="${RUNS_DIR}/tiny_stage1/ckpt_milestone.pt"
HORMONES_STD="${RUNS_DIR}/tiny_hormones.npy"

# Sanity check
if [ ! -f "${STAGE1_STD}" ]; then
    echo "ERROR: missing Stage-1 checkpoint at ${STAGE1_STD}"
    echo "  This script assumes /workspace/runs/tiny_stage1/ckpt_milestone.pt exists"
    echo "  (transferred from the previous rig). Run the rsync from your old rig first."
    exit 1
fi
if [ ! -f "${HORMONES_STD}" ]; then
    echo "ERROR: missing hormones bank at ${HORMONES_STD}"
    echo "  Transfer /workspace/runs/tiny_hormones.npy from your previous rig."
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
echo "Hormones bank reused: ${HORMONES_STD}"

# =============================================================================
# Step 1 — IA: continue from Stage-1 std with IA enabled (gate zero-init)
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
# Step 2 — Both: continue from Stage-1 std with IA + extracted hormones
# =============================================================================
BOTH_DIR="${RUNS_DIR}/tiny_both"
if [ ! -f "${BOTH_DIR}/ckpt_final.pt" ]; then
    log "Both Stage-2 (continuation from Stage-1 std, IA + hormone routing)"
    ${LAUNCH} scripts/train.py ${COMMON} \
        --intra_attn --ia_stride ${IA_STRIDE} \
        --hormones --hormone_path "${HORMONES_STD}" \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_STD}" \
        --out "${BOTH_DIR}"
else
    log "Both Stage-2: ckpt_final.pt found, skipping"
fi

# =============================================================================
# Done
# =============================================================================
log "ALL IA TRAINING DONE"
for d in tiny_ia tiny_both; do
    if [ -f "${RUNS_DIR}/${d}/ckpt_final.pt" ]; then
        v=$(grep -oP '"val_loss":\s*\K[\d.]+' ${RUNS_DIR}/${d}/train.jsonl 2>/dev/null | tail -1)
        echo "  ✓ ${d}  final val_loss=${v}"
    else
        echo "  ✗ ${d}  MISSING"
    fi
done

echo ""
echo "Next: run evals on the new IA + Both checkpoints"
echo "  Update VARIANTS array in scripts/run_all_evals_parallel.sh to include:"
echo "    \"tiny_ia:0:1\""
echo "    \"tiny_both:1:1\""
echo "  then: bash scripts/run_all_evals_parallel.sh"
