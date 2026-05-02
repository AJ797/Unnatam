#!/usr/bin/env bash
# Run zero-shot evals on all trained variants IN PARALLEL across GPUs.
# Each variant pinned to one GPU (eval is tiny — ~1GB VRAM).
# 5 variants × 1 GPU each → finishes in ~1h instead of ~5h serial.
#
# Usage:
#   bash scripts/run_all_evals_parallel.sh
#   QUICK=1 bash scripts/run_all_evals_parallel.sh   # 200-example smoke
#   FORCE=1 bash scripts/run_all_evals_parallel.sh   # redo even if evals.json exists

set -e
source /venv/main/bin/activate
cd /workspace/Unnatam

RUNS=/workspace/runs
QUICK_FLAG=""
[ "${QUICK:-0}" = "1" ] && QUICK_FLAG="--quick" && echo "QUICK mode (200 examples/benchmark)"

# (variant_name, has_hormones, has_intra_attn)
VARIANTS=(
    "tiny_base:0:0"
    "tiny_hr:1:0"
    "tiny_hr_rand:1:0"
    "tiny_hr_fixedgate:1:0"
    "tiny_hr_noext:1:0"
    # Add IA variants here once trained:
    # "tiny_ia:0:1"
    # "tiny_both:1:1"
)

mkdir -p /workspace/eval_logs
PIDS=()
NAMES=()
GPU=0

for entry in "${VARIANTS[@]}"; do
    IFS=":" read -r name has_hr has_ia <<< "${entry}"
    ckpt="${RUNS}/${name}/ckpt_final.pt"
    out="${RUNS}/${name}/evals.json"

    if [ ! -f "${ckpt}" ]; then
        echo "  ⚠ skipping ${name}: ckpt missing"
        continue
    fi
    if [ -f "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "  ✓ skipping ${name}: evals.json already exists"
        continue
    fi

    flags=""
    [ "${has_hr}" = "1" ] && flags="${flags} --use_hormones"
    [ "${has_ia}" = "1" ] && flags="${flags} --use_intra_attn"

    log="/workspace/eval_logs/${name}.log"
    echo "Launching ${name} on GPU ${GPU} → ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} nohup python3 scripts/run_evals.py \
        --ckpt "${ckpt}" --size tiny --vocab_size 50257 \
        --out "${out}" \
        ${flags} ${QUICK_FLAG} \
        > "${log}" 2>&1 < /dev/null &
    PIDS+=($!)
    NAMES+=("${name}")
    GPU=$((GPU + 1))

    # Stagger launches by 5s so HF dataset cache doesn't get hammered
    sleep 5
done

if [ ${#PIDS[@]} -eq 0 ]; then
    echo ""
    echo "Nothing to do (all evals.json present, set FORCE=1 to redo)"
    exit 0
fi

echo ""
echo "================================================"
echo "  ${#PIDS[@]} eval jobs launched in parallel"
echo "================================================"
echo "Monitor with:"
echo "  tail -f /workspace/eval_logs/<variant>.log"
echo "  watch -n 30 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo ""
echo "Waiting for all to finish..."
echo ""

# Wait for all jobs and report
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if wait "${pid}"; then
        echo "  ✓ ${name} done"
    else
        echo "  ✗ ${name} FAILED — see /workspace/eval_logs/${name}.log"
    fi
done

echo ""
echo "================================================"
echo "  ALL EVALS DONE"
echo "================================================"
for entry in "${VARIANTS[@]}"; do
    IFS=":" read -r name _ _ <<< "${entry}"
    out="${RUNS}/${name}/evals.json"
    if [ -f "${out}" ]; then
        echo ""
        echo "  ${name}:"
        python3 -c "
import json
with open('${out}') as f:
    d = json.load(f)
for r in d['results']:
    print(f\"    {r['name']:<15s} {r['metric']:<10s} {r['value']:.4f}  (n={r['n_examples']})\")
"
    fi
done
