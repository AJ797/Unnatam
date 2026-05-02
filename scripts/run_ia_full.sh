#!/usr/bin/env bash
# Full IA training pipeline, designed for a high-VRAM rig (e.g. 8× 96GB).
# Runs three IA-using variants:
#   1. Stage-1 IA (fresh, 1B tokens, intra_attn + no hormones)
#   2. Stage-2 IA (continuation, 1B more tokens, no hormones)
#   3. Stage-2 Both (continuation from Stage-1 IA, IA + extracted hormones)
#
# Tuned for:
#   - 96GB VRAM per GPU → micro_batch=16 (vs 2 on 24GB rigs)
#   - Grad checkpointing OFF (saves the doubled Python loop tax)
#   - ia_stride=32 (slot interaction every 32 scan steps, not every step)
#
# These three variants give us the IA, Both, and IA-Stage-1 row of the table.
#
# Usage:
#   bash scripts/run_ia_full.sh
#   STAGE1_TOKENS=1B STAGE2_TOKENS=1B IA_STRIDE=32 MICRO_BATCH=16 GRAD_ACCUM=1 \
#     bash scripts/run_ia_full.sh

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
GPUS="${GPUS:-8}"
STAGE1_TOKENS="${STAGE1_TOKENS:-1B}"
STAGE2_TOKENS="${STAGE2_TOKENS:-1B}"
MICRO_BATCH="${MICRO_BATCH:-16}"   # 96GB VRAM lets us use much bigger batches
GRAD_ACCUM="${GRAD_ACCUM:-1}"      # mb*ga*gpus*1024 = effective batch ≈ 131K tokens
IA_STRIDE="${IA_STRIDE:-32}"
WARMUP="${WARMUP:-2000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
# ──────────────────────────────────────────────────────────────────────────────

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

# =============================================================================
# Step 1 — Stage-1 IA (1B tokens, fresh, intra_attn ON)
# =============================================================================
STAGE1_IA_DIR="${RUNS_DIR}/tiny_ia_stage1"
if [ ! -f "${STAGE1_IA_DIR}/ckpt_milestone.pt" ]; then
    log "STAGE 1 IA (intracellular scan, ${STAGE1_TOKENS} tokens)"
    ${LAUNCH} scripts/train.py ${COMMON} \
        --intra_attn --ia_stride ${IA_STRIDE} \
        --tokens "${STAGE1_TOKENS}" \
        --milestone_tokens "${STAGE1_TOKENS}" \
        --out "${STAGE1_IA_DIR}"
else
    log "STAGE 1 IA: ckpt_milestone.pt found, skipping"
fi

# =============================================================================
# Step 2 — Extract IA hormones from Stage-1 IA checkpoint
# =============================================================================
HORMONES_IA="${RUNS_DIR}/tiny_hormones_ia.npy"
if [ ! -f "${HORMONES_IA}" ]; then
    log "EXTRACTING hormones from Stage-1 IA → ${HORMONES_IA}"
    python3 scripts/extract_hormones.py \
        --ckpt "${STAGE1_IA_DIR}/ckpt_milestone.pt" \
        --size tiny --vocab_size 50257 --intra_attn \
        --out "${HORMONES_IA}"
else
    log "Hormones IA: already extracted, skipping"
fi

# =============================================================================
# Step 3 — Stage-2 IA (continuation, 1B more, no hormones)
# =============================================================================
IA_DIR="${RUNS_DIR}/tiny_ia"
if [ ! -f "${IA_DIR}/ckpt_final.pt" ]; then
    log "IA Stage-2 (continuation, ${STAGE2_TOKENS} tokens)"
    ${LAUNCH} scripts/train.py ${COMMON} \
        --intra_attn --ia_stride ${IA_STRIDE} \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_IA_DIR}/ckpt_milestone.pt" \
        --out "${IA_DIR}"
else
    log "IA Stage-2: ckpt_final.pt found, skipping"
fi

# =============================================================================
# Step 4 — Stage-2 Both (IA + extracted hormones, ${STAGE2_TOKENS})
# =============================================================================
BOTH_DIR="${RUNS_DIR}/tiny_both"
if [ ! -f "${BOTH_DIR}/ckpt_final.pt" ]; then
    log "Both Stage-2 (continuation, IA + hormone routing)"
    ${LAUNCH} scripts/train.py ${COMMON} \
        --intra_attn --ia_stride ${IA_STRIDE} \
        --hormones --hormone_path "${HORMONES_IA}" \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_IA_DIR}/ckpt_milestone.pt" \
        --out "${BOTH_DIR}"
else
    log "Both Stage-2: ckpt_final.pt found, skipping"
fi

# =============================================================================
# Done
# =============================================================================
log "ALL IA TRAINING DONE"
for d in tiny_ia_stage1 tiny_ia tiny_both; do
    if [ -f "${RUNS_DIR}/${d}/ckpt_final.pt" ] || [ -f "${RUNS_DIR}/${d}/ckpt_milestone.pt" ]; then
        v=$(grep -oP '"val_loss":\s*\K[\d.]+' ${RUNS_DIR}/${d}/train.jsonl 2>/dev/null | tail -1)
        echo "  ✓ ${d}  final val_loss=${v}"
    else
        echo "  ✗ ${d}  MISSING"
    fi
done

echo ""
echo "Next: run evals on the new variants"
echo "  bash scripts/run_all_evals.sh    # update VARIANTS array first if needed"
