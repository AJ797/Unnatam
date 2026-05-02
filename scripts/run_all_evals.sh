#!/usr/bin/env bash
# Run zero-shot evals on all trained variants.
# Each variant takes ~30-60 min on a single GPU for the full 5-benchmark sweep.
# Runs sequentially on cuda:0 to avoid contention.
#
# Usage:
#   bash scripts/run_all_evals.sh           # full evals
#   QUICK=1 bash scripts/run_all_evals.sh   # 200-example smoke test (~5 min/variant)

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

RUNS=/workspace/runs
QUICK_FLAG=""
if [ "${QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
    echo "QUICK mode: subsampling each benchmark to 200 examples"
fi

# (variant_name, has_hormones, has_intra_attn)
VARIANTS=(
    "tiny_base:0:0"
    "tiny_hr:1:0"
    "tiny_hr_rand:1:0"
    "tiny_hr_fixedgate:1:0"
    "tiny_hr_noext:1:0"
)

for entry in "${VARIANTS[@]}"; do
    IFS=":" read -r name has_hr has_ia <<< "${entry}"
    ckpt="${RUNS}/${name}/ckpt_final.pt"
    out="${RUNS}/${name}/evals.json"

    if [ ! -f "${ckpt}" ]; then
        echo "  ⚠ skipping ${name}: ckpt missing"
        continue
    fi
    if [ -f "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "  ✓ skipping ${name}: evals.json already exists (set FORCE=1 to redo)"
        continue
    fi

    flags=""
    [ "${has_hr}" = "1" ] && flags="${flags} --use_hormones"
    [ "${has_ia}" = "1" ] && flags="${flags} --use_intra_attn"

    echo ""
    echo "================================================"
    echo "  EVAL: ${name}"
    echo "================================================"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_evals.py \
        --ckpt "${ckpt}" --size tiny --vocab_size 50257 \
        --out "${out}" \
        ${flags} ${QUICK_FLAG}
done

echo ""
echo "================================================"
echo "  ALL EVALS DONE"
echo "================================================"
for entry in "${VARIANTS[@]}"; do
    IFS=":" read -r name _ _ <<< "${entry}"
    out="${RUNS}/${name}/evals.json"
    if [ -f "${out}" ]; then
        echo "  ✓ ${name}: ${out}"
    else
        echo "  ✗ ${name}: MISSING"
    fi
done
