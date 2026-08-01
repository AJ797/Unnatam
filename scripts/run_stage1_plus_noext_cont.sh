#!/usr/bin/env bash
# Compound overnight script for Reviewer D15v's HR-noext-cont isolation ablation.
#
# The original tiny_stage1/ckpt_milestone.pt was not preserved locally, so this
# script retrains Stage 1 from scratch and then runs the isolation experiment.
#
# Pipeline:
#   1. Stage 1: train tiny hybrid from scratch for 1B tokens (~11h)
#   2. Extract random-init hormones (already have tiny_hormones_noext.npy if
#      restored from artifactsall.tgz — otherwise re-extract from init)
#   3. Stage 2 continuation with the random-init-extracted bank (~11h)
#   4. Tarball artefacts to /workspace/artefacts/ for safekeeping
#
# Total: ~22h wall-clock, ~$30 on 8×RTX 4090
#
# Interpretation
# --------------
# If HR-noext-cont's final val PPL ≈ 26.13 (Base): schedule causes the damage
#   observed in HR-noext; bank content is neutral in a staged continuation.
# If HR-noext-cont's final val PPL ≈ 28.20 (HR-noext): bank content also matters;
#   we soften the "schedule is the loss-relevant axis" claim in the paper.
# In between: both factors contribute; nuanced revision.
#
# Usage:
#   PYTHONUNBUFFERED=1 nohup bash scripts/run_stage1_plus_noext_cont.sh \
#     > /workspace/noext_cont.log 2>&1 &
#   disown

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
ART_DIR="${ART_DIR:-/workspace/artefacts_noext_cont}"
GPUS="${GPUS:-8}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

STAGE1_DIR="${RUNS_DIR}/tiny_stage1"
NOEXT_CONT_DIR="${RUNS_DIR}/tiny_hr_noext_cont"
HORMONES_NOEXT="${RUNS_DIR}/tiny_hormones_noext.npy"

mkdir -p "${ART_DIR}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() {
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  [$(stamp)]  $*"
    echo "═══════════════════════════════════════════════"
    echo ""
}

cleanup_gpus() {
    pkill -9 -f "train.py" 2>/dev/null
    pkill -9 -f "torchrun" 2>/dev/null
    sleep 5
    for i in $(seq 1 30); do
        max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
        max_mem=${max_mem:-9999}
        [ "${max_mem}" -lt 500 ] && return 0
        sleep 1
    done
    echo "  WARN: GPUs still busy after 30s"
}

export PYTHONUNBUFFERED=1

log "OVERNIGHT: Stage 1 retrain + HR-noext-cont ablation (~22h)"

# ============================================================================
# Phase 1 — Stage 1 (~11h, ~$15)
# ============================================================================
if [ -f "${STAGE1_DIR}/ckpt_milestone.pt" ] || [ -f "${STAGE1_DIR}/ckpt_final.pt" ]; then
    log "Stage 1 checkpoint already exists — skipping retrain"
else
    log "PHASE 1 — Stage 1 (1B tokens, no hormones, ~11h)"
    rm -rf "${STAGE1_DIR}"
    timeout --kill-after=60 43200 torchrun --nproc_per_node=${GPUS} --master_port=29500 \
        scripts/train.py \
        --size tiny --vocab_size 50257 \
        --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val \
        --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM} \
        --warmup_steps 2000 --ckpt_interval 5000 \
        --log_interval 100 --eval_interval 1000 \
        --tokens 1B --milestone_tokens 1B \
        --out "${STAGE1_DIR}"
    rc=$?
    log "Stage 1 exit code: ${rc}"

    # Sanity: promote ckpt_final to ckpt_milestone if milestone missing
    if [ ! -f "${STAGE1_DIR}/ckpt_milestone.pt" ] && [ -f "${STAGE1_DIR}/ckpt_final.pt" ]; then
        cp "${STAGE1_DIR}/ckpt_final.pt" "${STAGE1_DIR}/ckpt_milestone.pt"
    fi

    # Back up stage 1 immediately
    log "Backing up Stage 1"
    tar czf "${ART_DIR}/stage1.tgz" -C "${RUNS_DIR}" "tiny_stage1"
    ls -lh "${ART_DIR}/stage1.tgz"
fi

if [ ! -f "${STAGE1_DIR}/ckpt_milestone.pt" ]; then
    log "ABORT: Stage 1 failed, no ckpt_milestone.pt"
    exit 1
fi

# ============================================================================
# Phase 2 — extract random-init hormones (~30s)
# ============================================================================
if [ -f "${HORMONES_NOEXT}" ]; then
    log "Random-init hormones already exist at ${HORMONES_NOEXT} — skipping"
else
    log "PHASE 2 — extract hormones from a FRESHLY-INITIALISED (untrained) model"
    python3 scripts/extract_hormones.py \
        --init \
        --size tiny --vocab_size 50257 \
        --out "${HORMONES_NOEXT}"
    ls -lh "${HORMONES_NOEXT}"
fi

# ============================================================================
# Cleanup between phases
# ============================================================================
log "Cleanup between phases — killing stragglers, 2 min wait"
cleanup_gpus
sleep 120

# ============================================================================
# Phase 3 — HR-noext-cont Stage 2 (~11h, ~$15)
# ============================================================================
if [ -f "${NOEXT_CONT_DIR}/ckpt_final.pt" ]; then
    log "HR-noext-cont already done — skipping"
else
    log "PHASE 3 — HR-noext-cont Stage 2 continuation (1B tokens, ~11h)"
    rm -rf "${NOEXT_CONT_DIR}"
    timeout --kill-after=60 43200 torchrun --nproc_per_node=${GPUS} --master_port=29501 \
        scripts/train.py \
        --size tiny --vocab_size 50257 \
        --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val \
        --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM} \
        --warmup_steps 2000 --ckpt_interval 5000 \
        --log_interval 100 --eval_interval 1000 \
        --hormones --hormone_path "${HORMONES_NOEXT}" \
        --tokens 1B \
        --resume "${STAGE1_DIR}/ckpt_milestone.pt" \
        --out "${NOEXT_CONT_DIR}"
    rc=$?
    log "HR-noext-cont exit code: ${rc}"
fi

# ============================================================================
# Phase 4 — Backup + eval
# ============================================================================
if [ -f "${NOEXT_CONT_DIR}/ckpt_final.pt" ]; then
    log "PHASE 4 — Backup HR-noext-cont"
    tar czf "${ART_DIR}/hr_noext_cont.tgz" -C "${RUNS_DIR}" "tiny_hr_noext_cont"
    ls -lh "${ART_DIR}/hr_noext_cont.tgz"

    log "PHASE 4 — Zero-shot evals (~1h on single GPU)"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_evals.py \
        --ckpt "${NOEXT_CONT_DIR}/ckpt_final.pt" \
        --size tiny --vocab_size 50257 --use_hormones \
        --out "${NOEXT_CONT_DIR}/evals.json"

    cp "${NOEXT_CONT_DIR}/evals.json" "${ART_DIR}/hr_noext_cont_evals.json"
    cp "${NOEXT_CONT_DIR}/train.jsonl" "${ART_DIR}/hr_noext_cont_train.jsonl"

    log "Final combined tarball"
    tar czf "${ART_DIR}/all.tgz" -C "${RUNS_DIR}" "tiny_stage1" "tiny_hr_noext_cont"
    ls -lh "${ART_DIR}/"
fi

log "OVERNIGHT COMPLETE"
v=$(grep -oP '"val_loss":\s*\K[\d.]+' "${NOEXT_CONT_DIR}/train.jsonl" 2>/dev/null | tail -1)
ppl=$(python3 -c "import math; print(f'{math.exp(${v:-0}):.3f}')" 2>/dev/null)
echo "  HR-noext-cont  final val_loss=${v}  PPL=${ppl}"
echo ""
echo "Reference points:"
echo "  Base           PPL 26.13  ← baseline"
echo "  HR-noext       PPL 28.20  ← from-scratch with random-init hormones"
echo "  HR-noext-cont  PPL ${ppl}  ← STAGED with random-init hormones"
echo ""
echo "Interpretation:"
echo "  ≈ 26.13 → schedule confirmed as causal variable"
echo "  ≈ 28.20 → bank content also damages"
echo "  in between → both factors contribute"
