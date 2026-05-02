#!/usr/bin/env bash
# Overnight orchestrator: IA training → cleanup → HR-forced training, with
# defensive backups at every stage. Designed to survive an unattended overnight
# run where the user might oversleep or the rig might run out of credit.
#
# What it does, in order:
#   1. (~8.5h) IA Stage-2 training, 20M tokens, warmup=50
#   2. Tarball IA artefacts → /workspace/artefacts_overnight/ia.tgz
#   3. Kill any lingering training procs, wait 2 min for GPU memory to clear
#   4. (~5h)   HR-forced Stage-2 training, 200M tokens, init_gate=0.2, alpha=3
#   5. Tarball HR-forced artefacts → /workspace/artefacts_overnight/hr_forced.tgz
#   6. Combined tar of both → /workspace/artefacts_overnight/all_overnight.tgz
#
# Survives:
#   - SSH disconnect (uses nohup at launch)
#   - Logger buffering (PYTHONUNBUFFERED=1)
#   - One run hanging (10h timeout on IA, 7h timeout on HR-forced)
#   - One run crashing (HR-forced runs even if IA failed; both are independent
#     of each other since both resume from tiny_stage1, not from each other)
#   - Rig shutdown mid-HR-forced (IA artefacts already tarred)
#
# Usage:
#   PYTHONUNBUFFERED=1 nohup bash scripts/run_overnight.sh \
#     > /workspace/overnight.log 2>&1 &
#   disown
#
# Then check:
#   tail -f /workspace/overnight.log
#   ls -lh /workspace/artefacts_overnight/

# Intentionally NOT using `set -e` — we want explicit error handling so a
# failure in one phase doesn't prevent the backup of the other phase's data.

source /venv/main/bin/activate
cd /workspace/Unnatam

ART_DIR=/workspace/artefacts_overnight
mkdir -p "${ART_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  [$(stamp)]  $*"
    echo "═══════════════════════════════════════════════"
    echo ""
}

cleanup_gpus() {
    # Kill any lingering training processes; wait for GPU memory to free.
    pkill -9 -f "train.py" 2>/dev/null
    pkill -9 -f "torchrun" 2>/dev/null
    sleep 5
    for i in $(seq 1 30); do
        max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
        max_mem=${max_mem:-9999}
        if [ "${max_mem}" -lt 500 ]; then
            echo "  [$(stamp)] GPUs cleared (max ${max_mem} MiB across cards)"
            return 0
        fi
        sleep 1
    done
    echo "  [$(stamp)] WARN: at least one GPU still reports memory usage after 30s"
    return 1
}

backup_run() {
    # backup_run <runs/ subdir name> <output tarball name without extension>
    local dir=$1
    local name=$2
    if [ -d "/workspace/runs/${dir}" ]; then
        echo "  [$(stamp)] backing up runs/${dir} → ${ART_DIR}/${name}.tgz"
        tar czf "${ART_DIR}/${name}.tgz" -C /workspace/runs "${dir}" 2>&1 | tail -3
        ls -lh "${ART_DIR}/${name}.tgz"
    else
        echo "  [$(stamp)] runs/${dir} not found, skipping backup"
    fi
}

export PYTHONUNBUFFERED=1

log "OVERNIGHT RUN START — total expected wall-clock ~13.5h"
echo "  Today's date: $(date '+%Y-%m-%d')"
echo "  Disk free:"
df -h /workspace | tail -1
echo "  Existing runs:"
ls -1 /workspace/runs/ 2>/dev/null
echo ""

# ============================================================================
# Phase 1 — IA training (~8.5h)
# ============================================================================
log "PHASE 1 — IA Stage-2 training (warmup=50, 20M tokens)"

rm -rf /workspace/runs/tiny_ia

# Hard 10h timeout (should finish in ~8.5h, 90 min headroom for any slow step).
# --kill-after=60 sends SIGKILL 60s after SIGTERM if the run doesn't clean up.
export STAGE2_TOKENS=20M
export MICRO_BATCH=2
export GRAD_ACCUM=4
export IA_STRIDE=32
export LOG_INTERVAL=10
export WARMUP=50
timeout --kill-after=60 36000 bash scripts/run_ia_full.sh
ia_rc=$?
log "IA exit code: ${ia_rc}  ($([ ${ia_rc} -eq 0 ] && echo "success" || echo "non-zero — check log"))"

# Backup whatever exists, even if rc != 0 — partial state is better than nothing.
if [ -f /workspace/runs/tiny_ia/ckpt_final.pt ]; then
    log "PHASE 1 — backup IA"
    backup_run "tiny_ia" "ia"
else
    log "PHASE 1 — no ckpt_final.pt; backing up partial state if any"
    backup_run "tiny_ia" "ia_partial"
fi

# ============================================================================
# Cleanup between runs
# ============================================================================
log "Cleanup — killing stragglers, waiting 2 minutes for GPU memory to free"
cleanup_gpus
sleep 120
echo "  [$(stamp)] post-sleep GPU state:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# Clear env vars that were set for IA but might leak into HR-forced (defensive).
unset STAGE2_TOKENS MICRO_BATCH GRAD_ACCUM IA_STRIDE LOG_INTERVAL WARMUP

# ============================================================================
# Phase 2 — HR-forced training (~5h)
# ============================================================================
# HR-forced is independent of IA — both resume from tiny_stage1, not from each
# other — so we run it regardless of whether IA succeeded.
log "PHASE 2 — HR-forced Stage-2 training (init_gate=0.2, alpha=3.0, 200M tokens)"

rm -rf /workspace/runs/tiny_hr_forced

# Hard 7h timeout (should finish in ~5h, 2h headroom).
export STAGE2_TOKENS=200M
export MICRO_BATCH=8
export GRAD_ACCUM=1
export INIT_GATE=0.2
export ALPHA=3.0
timeout --kill-after=60 25200 bash scripts/run_hr_forced.sh
hf_rc=$?
log "HR-forced exit code: ${hf_rc}  ($([ ${hf_rc} -eq 0 ] && echo "success" || echo "non-zero — check log"))"

if [ -f /workspace/runs/tiny_hr_forced/ckpt_final.pt ]; then
    log "PHASE 2 — backup HR-forced"
    backup_run "tiny_hr_forced" "hr_forced"
else
    log "PHASE 2 — no ckpt_final.pt; backing up partial state if any"
    backup_run "tiny_hr_forced" "hr_forced_partial"
fi

# ============================================================================
# Final combined backup + summary
# ============================================================================
log "FINAL — combined backup"
COMBINED=()
[ -d /workspace/runs/tiny_ia ] && COMBINED+=("tiny_ia")
[ -d /workspace/runs/tiny_hr_forced ] && COMBINED+=("tiny_hr_forced")
if [ ${#COMBINED[@]} -gt 0 ]; then
    tar czf "${ART_DIR}/all_overnight.tgz" -C /workspace/runs "${COMBINED[@]}" 2>&1 | tail -3
    ls -lh "${ART_DIR}/all_overnight.tgz"
fi

log "OVERNIGHT RUN COMPLETE"
echo "  IA exit:        ${ia_rc}"
echo "  HR-forced exit: ${hf_rc}"
echo "  Artefacts in:   ${ART_DIR}/"
ls -lh "${ART_DIR}/"
echo ""
echo "  Final val losses:"
if [ -f /workspace/runs/tiny_ia/train.jsonl ]; then
    v=$(grep -oP '"val_loss":\s*\K[\d.]+' /workspace/runs/tiny_ia/train.jsonl | tail -1)
    echo "    tiny_ia          last_val_loss=${v}  ppl=$(python3 -c "import math;print(f'{math.exp(${v:-0}):.3f}')" 2>/dev/null)"
fi
if [ -f /workspace/runs/tiny_hr_forced/train.jsonl ]; then
    v=$(grep -oP '"val_loss":\s*\K[\d.]+' /workspace/runs/tiny_hr_forced/train.jsonl | tail -1)
    echo "    tiny_hr_forced   last_val_loss=${v}  ppl=$(python3 -c "import math;print(f'{math.exp(${v:-0}):.3f}')" 2>/dev/null)"
fi

log "Done. Sleep tight."
