#!/usr/bin/env bash
# HR-forced ablation: train HR with a "warm" gate initialisation and an alpha
# multiplier on the injection, to test the hypothesis that the original HR was
# functionally a no-op because the perturbation magnitude never grew large
# enough to influence the loss objective at this scale.
#
# What this controls:
#   - init_gate=0.2  → the gate starts engaged, not at zero (vs HR's 0.0)
#   - alpha=3        → the gated injection is 3× amplified before adding to residual
#                      (effective initial perturbation ≈ 0.2 × 3 × ||shift|| = 0.6 × ||shift||,
#                       which is roughly 3× larger than HR's late-training peak of ~0.23)
#   - hormone bank   → the unit-norm extracted bank from Stage 1 std (the F.normalize
#                      fix in unnatam.py ensures this is actually unit norm now)
#
# Outcomes:
#   - val_loss < base  → the mechanism CAN help when forced to engage; small models
#                        chose not to engage on their own → "capacity-bottlenecked routing"
#                        finding is real
#   - val_loss = base  → the model adapts to absorb even forced injection;
#                        "representation-level organisation independent of objective"
#                        finding is the strongest version
#   - val_loss > base  → forced engagement actively interferes; mechanism is
#                        bounded only when softly initialised
#
# Cost: ~5 hours on 8× RTX 5090 at ~12,000 tok/s (fused kernel, no IA), ~$15-20.
#
# Usage:
#   bash scripts/run_hr_forced.sh
#   STAGE2_TOKENS=500M INIT_GATE=0.2 ALPHA=3.0 bash scripts/run_hr_forced.sh

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
GPUS="${GPUS:-8}"
STAGE2_TOKENS="${STAGE2_TOKENS:-200M}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
INIT_GATE="${INIT_GATE:-0.2}"
ALPHA="${ALPHA:-3.0}"
WARMUP="${WARMUP:-200}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"

STAGE1_STD="${RUNS_DIR}/tiny_stage1/ckpt_milestone.pt"
HORMONES_STD="${RUNS_DIR}/tiny_hormones.npy"

if [ ! -f "${STAGE1_STD}" ]; then echo "ERROR: missing ${STAGE1_STD}"; exit 1; fi
if [ ! -f "${HORMONES_STD}" ]; then echo "ERROR: missing ${HORMONES_STD}"; exit 1; fi

OUT_DIR="${RUNS_DIR}/tiny_hr_forced"
EFF=$((MICRO_BATCH * GRAD_ACCUM * GPUS * 1024))
echo "═══════════════════════════════════════════════"
echo "  HR-forced  init_gate=${INIT_GATE}  alpha=${ALPHA}  ${STAGE2_TOKENS} tokens"
echo "  Effective batch ≈ ${EFF} tokens/opt step"
echo "═══════════════════════════════════════════════"

if [ -f "${OUT_DIR}/ckpt_final.pt" ]; then
    echo "ckpt_final.pt exists — skipping. Delete the dir to re-run."
    exit 0
fi

torchrun --nproc_per_node=${GPUS} --master_port=29501 scripts/train.py \
    --size tiny --vocab_size 50257 \
    --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val \
    --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM} \
    --warmup_steps ${WARMUP} --ckpt_interval ${CKPT_INTERVAL} \
    --log_interval ${LOG_INTERVAL} --eval_interval ${EVAL_INTERVAL} \
    --hormones --hormone_path "${HORMONES_STD}" \
    --init_gate ${INIT_GATE} --alpha ${ALPHA} \
    --tokens "${STAGE2_TOKENS}" \
    --resume "${STAGE1_STD}" \
    --out "${OUT_DIR}"

echo ""
echo "═══════════════════════════════════════════════"
echo "  HR-forced DONE"
echo "═══════════════════════════════════════════════"
v=$(grep -oP '"val_loss":\s*\K[\d.]+' ${OUT_DIR}/train.jsonl 2>/dev/null | tail -1)
echo "  final val_loss=${v}"
echo ""
echo "Compare:"
echo "  Base    final val_loss=3.2632 (PPL=26.13)"
echo "  HR      final val_loss=3.2632 (PPL=26.13)"
echo "  HR-forced final val_loss=${v}  PPL=$(python3 -c "import math;print(f'{math.exp(${v:-0}):.3f}')")"
echo ""
echo "Then: bash scripts/inspect_ia_gates.sh CKPT=${OUT_DIR}/ckpt_final.pt   # gates"
echo "      python3 scripts/run_evals.py --ckpt ${OUT_DIR}/ckpt_final.pt --size tiny --vocab_size 50257 --use_hormones --out ${OUT_DIR}/evals.json"
