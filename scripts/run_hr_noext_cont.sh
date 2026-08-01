#!/usr/bin/env bash
# HR-noext-cont ablation: isolation experiment for reviewer Change 1.
#
# Motivation
# ----------
# HR-noext confounds three variables: (i) vectors extracted from a randomly-
# initialised model, (ii) trained from scratch rather than as a Stage-2
# continuation, and (iii) 2B contiguous budget vs 1B+1B staged. We attribute
# the loss damage to (ii/iii) — the schedule — because HR-rand, which uses
# equally-uninformative random unit vectors, is benign as a continuation.
# Reviewer D15v flagged that a clean isolation had not been run.
#
# This variant tests the schedule attribution:
#   staged Stage-2 continuation from tiny_stage1/ckpt_milestone.pt
#   + random-init-extracted hormone bank (tiny_hormones_noext.npy)
#
# Interpretation
# --------------
#   val PPL ≈ 26.13 (Base)   → schedule IS the causal variable; the bank
#                              content is neutral when introduced as a
#                              Stage-2 continuation. HR-noext's damage is
#                              from the from-scratch schedule.
#   val PPL ≈ 28.20 (noext)  → bank content ALSO matters. The claim that
#                              schedule alone drives the damage is wrong.
#   val PPL in between       → both matter to some degree; the story becomes
#                              "schedule dominates but content contributes."
#
# Wall-clock estimate on 8× RTX 4090:
#   1B Stage-2 tokens at ~12,800 tok/s ≈ 22 hours ≈ $25-30 at $1.20-1.35/GPU-hr
#
# Reuses:
#   /workspace/runs/tiny_stage1/ckpt_milestone.pt   (existing Stage-1 std)
#   /workspace/runs/tiny_hormones_noext.npy         (existing bank from random init)
#
# Usage:
#   bash scripts/run_hr_noext_cont.sh
#   STAGE2_TOKENS=1B MICRO_BATCH=8 GRAD_ACCUM=2 bash scripts/run_hr_noext_cont.sh

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
GPUS="${GPUS:-8}"
STAGE2_TOKENS="${STAGE2_TOKENS:-1B}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WARMUP="${WARMUP:-2000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"

STAGE1_STD="${RUNS_DIR}/tiny_stage1/ckpt_milestone.pt"
HORMONES_NOEXT="${RUNS_DIR}/tiny_hormones_noext.npy"

if [ ! -f "${STAGE1_STD}" ]; then
    echo "ERROR: missing Stage-1 checkpoint at ${STAGE1_STD}"
    echo "  This variant reuses the Stage-1 std checkpoint from the original ablation run."
    exit 1
fi
if [ ! -f "${HORMONES_NOEXT}" ]; then
    echo "ERROR: missing hormones bank at ${HORMONES_NOEXT}"
    echo "  Should already exist from the HR-noext extraction step of the original ablation."
    exit 1
fi

OUT_DIR="${RUNS_DIR}/tiny_hr_noext_cont"
EFF=$((MICRO_BATCH * GRAD_ACCUM * GPUS * 1024))

echo "═══════════════════════════════════════════════"
echo "  HR-noext-cont (isolation ablation)"
echo "  Stage-1 reused:   ${STAGE1_STD}"
echo "  Hormones bank:    ${HORMONES_NOEXT}"
echo "  Stage-2 budget:   ${STAGE2_TOKENS}"
echo "  Effective batch:  ${EFF} tokens/opt step"
echo "═══════════════════════════════════════════════"

if [ -f "${OUT_DIR}/ckpt_final.pt" ]; then
    echo "ckpt_final.pt already exists — skipping. Delete the dir to re-run."
    exit 0
fi

export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=${GPUS} --master_port=29502 scripts/train.py \
    --size tiny --vocab_size 50257 \
    --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val \
    --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM} \
    --warmup_steps ${WARMUP} --ckpt_interval ${CKPT_INTERVAL} \
    --log_interval ${LOG_INTERVAL} --eval_interval ${EVAL_INTERVAL} \
    --hormones --hormone_path "${HORMONES_NOEXT}" \
    --tokens "${STAGE2_TOKENS}" \
    --resume "${STAGE1_STD}" \
    --out "${OUT_DIR}"

echo ""
echo "═══════════════════════════════════════════════"
echo "  HR-noext-cont DONE"
echo "═══════════════════════════════════════════════"

v=$(grep -oP '"val_loss":\s*\K[\d.]+' ${OUT_DIR}/train.jsonl 2>/dev/null | tail -1)
ppl=$(python3 -c "import math; print(f'{math.exp(${v:-0}):.3f}')" 2>/dev/null)

echo "  final val_loss=${v}  PPL=${ppl}"
echo ""
echo "Reference points:"
echo "  Base           final val_loss=3.2632  (PPL=26.13)   ← 1B+1B staged, NO hormones"
echo "  HR             final val_loss=3.2632  (PPL=26.13)   ← 1B+1B staged, extracted hormones"
echo "  HR-noext-cont  final val_loss=${v}  (PPL=${ppl})   ← 1B+1B STAGED, random-init hormones"
echo "  HR-noext       final val_loss=3.3393  (PPL=28.20)   ← 2B from scratch, random-init hormones"
echo ""
echo "Interpretation guide:"
echo "  If HR-noext-cont PPL ≈ 26.13 (Base)     → SCHEDULE causes damage. Bank content neutral in continuation."
echo "  If HR-noext-cont PPL ≈ 28.20 (HR-noext) → BANK CONTENT also damages. Schedule claim needs softening."
echo "  If somewhere in between                  → BOTH factors contribute."
echo ""
echo "Next steps:"
echo "  1. Inspect gates: CKPT=${OUT_DIR}/ckpt_final.pt bash scripts/inspect_ia_gates.sh"
echo "  2. Run evals:     CUDA_VISIBLE_DEVICES=0 python3 scripts/run_evals.py \\"
echo "                       --ckpt ${OUT_DIR}/ckpt_final.pt --size tiny --vocab_size 50257 \\"
echo "                       --use_hormones --out ${OUT_DIR}/evals.json"
