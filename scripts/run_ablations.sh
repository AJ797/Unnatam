#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh  —  Full ablation training for Unnatam (7 variants)
#
# Run on the vast.ai 4×4090 node after setup. Requires:
#   - tokenized data at $DATA_DIR/{train,val}/shard_*.bin
#   - conda env "unnatam" activated with mamba-ssm, causal-conv1d, bitsandbytes
#
# Usage:
#   bash scripts/run_ablations.sh            # tiny model ablations (~22h on 4×4090)
#   SIZE=small bash scripts/run_ablations.sh  # small model (USE WITH CAUTION — IA is slow)
#
# Variants produced:
#   Base          standard hybrid, no modifications
#   IA            intracellular attention only (ref scan — slower)
#   HR            hormone routing with extracted vectors (staged)
#   Both          IA + HR (staged)
#   HR-rand       hormone routing with random unit-norm vectors (staged)
#   HR-noext      hormone routing, vectors extracted at step 0
#   HR-fixedgate  hormone routing, gate=1.0 fixed (staged)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SIZE="${SIZE:-tiny}"
DATA_DIR="${DATA_DIR:-/workspace/data/fineweb}"
RUNS_DIR="${RUNS_DIR:-/workspace/runs}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
GPUS="${GPUS:-4}"
TOTAL_TOKENS="${TOTAL_TOKENS:-4B}"
STAGE1_TOKENS="${STAGE1_TOKENS:-1B}"
STAGE2_TOKENS="${STAGE2_TOKENS:-3B}"     # must equal TOTAL_TOKENS - STAGE1_TOKENS
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
# IA variants: try micro_batch=6 → 4 → 2 with auto-fallback on failure.
# grad_accum is matched so effective batch ≈ 131K tokens regardless of fallback.
# Format: "MB:GA" pairs, tried left-to-right.
IA_FALLBACK_CHAIN="${IA_FALLBACK_CHAIN:-6:5 4:8 2:16}"
SEQ_LEN="${SEQ_LEN:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
# ──────────────────────────────────────────────────────────────────────────────

LAUNCH="torchrun --nproc_per_node=${GPUS}"
TRAIN="scripts/train.py"
EXTRACT="scripts/extract_hormones.py"
COMMON_BASE="--size ${SIZE} --vocab_size ${VOCAB_SIZE}
        --data ${DATA_DIR}/train --val_data ${DATA_DIR}/val
        --seq_len ${SEQ_LEN} --warmup_steps ${WARMUP_STEPS}
        --ckpt_interval ${CKPT_INTERVAL} --log_interval ${LOG_INTERVAL}
        --eval_interval ${EVAL_INTERVAL}"

# Fast variants: standard batch
COMMON="${COMMON_BASE} --micro_batch_size ${MICRO_BATCH} --grad_accum_steps ${GRAD_ACCUM}"

# Run an IA training command with fallback through IA_FALLBACK_CHAIN. Resumes
# from any step checkpoint that was written before the crash.
# Args: $1 = output dir; $@ = remaining train.py flags (without micro_batch/grad_accum/--out)
run_ia_with_fallback() {
    local out_dir="$1"; shift
    local already_done="${out_dir}/ckpt_final.pt"
    local stage1_done="${out_dir}/ckpt_milestone.pt"

    # Already complete?
    if [ -f "${already_done}" ] || [ -f "${stage1_done}" ]; then
        # Caller will print the skip log
        return 0
    fi

    for ATTEMPT in ${IA_FALLBACK_CHAIN}; do
        local MB="${ATTEMPT%:*}"
        local GA="${ATTEMPT#*:}"

        # Find latest step checkpoint to resume from (if any, from a prior partial run)
        local resume_arg=""
        if [ -d "${out_dir}" ]; then
            local latest=$(ls "${out_dir}/" 2>/dev/null | grep -oE 'ckpt_step[0-9]+\.pt' | sort -V | tail -1)
            if [ -n "${latest}" ]; then
                resume_arg="--resume ${out_dir}/${latest}"
                echo "  ↻ resuming from ${latest}"
            fi
        fi

        echo ""
        echo "  ▶ trying micro_batch=${MB} grad_accum=${GA} (effective ≈ $((MB * GA * GPUS * SEQ_LEN)) tok/step)"
        ${LAUNCH} ${TRAIN} ${COMMON_BASE} \
            --micro_batch_size "${MB}" --grad_accum_steps "${GA}" \
            "$@" \
            ${resume_arg} \
            --out "${out_dir}"

        local rc=$?
        if [ ${rc} -eq 0 ]; then
            echo "  ✓ succeeded with micro_batch=${MB}"
            return 0
        fi
        echo "  ✗ exit ${rc} at micro_batch=${MB}; falling back…"
    done

    echo "  ✗ ALL fallbacks exhausted for ${out_dir}"
    return 1
}

mkdir -p "${RUNS_DIR}"

log() { echo ""; echo "═══════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════"; echo ""; }

# =============================================================================
# Step 0 — Shared Stage-1 checkpoints (standard scan, no hormones)
# =============================================================================

# ─── Stage-1 (standard scan) ─────────────────────────────────────────────────
STAGE1_DIR="${RUNS_DIR}/${SIZE}_stage1"
if [ ! -f "${STAGE1_DIR}/ckpt_milestone.pt" ]; then
    log "STAGE 1 (standard scan, ${STAGE1_TOKENS} tokens) → ${STAGE1_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --tokens "${STAGE1_TOKENS}" \
        --milestone_tokens "${STAGE1_TOKENS}" \
        --out "${STAGE1_DIR}"
else
    log "STAGE 1: checkpoint found, skipping → ${STAGE1_DIR}/ckpt_milestone.pt"
fi

# ─── Stage-1 IA (intracellular attention scan) ───────────────────────────────
STAGE1_IA_DIR="${RUNS_DIR}/${SIZE}_ia_stage1"
if [ ! -f "${STAGE1_IA_DIR}/ckpt_milestone.pt" ]; then
    log "STAGE 1 IA (intracellular scan, ${STAGE1_TOKENS} tokens) → ${STAGE1_IA_DIR}"
    run_ia_with_fallback "${STAGE1_IA_DIR}" \
        --intra_attn \
        --tokens "${STAGE1_TOKENS}" \
        --milestone_tokens "${STAGE1_TOKENS}"
else
    log "STAGE 1 IA: checkpoint found, skipping → ${STAGE1_IA_DIR}/ckpt_milestone.pt"
fi

# =============================================================================
# Step 1 — Hormone extraction from Stage-1 checkpoints
# =============================================================================

HORMONES_STD="${RUNS_DIR}/${SIZE}_hormones.npy"
if [ ! -f "${HORMONES_STD}" ]; then
    log "EXTRACTION (standard Stage-1 → ${HORMONES_STD})"
    python ${EXTRACT} \
        --ckpt "${STAGE1_DIR}/ckpt_milestone.pt" \
        --size "${SIZE}" --vocab_size "${VOCAB_SIZE}" \
        --out "${HORMONES_STD}"
else
    log "EXTRACTION: hormone bank found, skipping → ${HORMONES_STD}"
fi

HORMONES_IA="${RUNS_DIR}/${SIZE}_hormones_ia.npy"
if [ ! -f "${HORMONES_IA}" ]; then
    log "EXTRACTION IA (IA Stage-1 → ${HORMONES_IA})"
    python ${EXTRACT} \
        --ckpt "${STAGE1_IA_DIR}/ckpt_milestone.pt" \
        --size "${SIZE}" --vocab_size "${VOCAB_SIZE}" --intra_attn \
        --out "${HORMONES_IA}"
else
    log "EXTRACTION IA: hormone bank found, skipping → ${HORMONES_IA}"
fi

HORMONES_NOEXT="${RUNS_DIR}/${SIZE}_hormones_noext.npy"
if [ ! -f "${HORMONES_NOEXT}" ]; then
    log "EXTRACTION NOEXT (untrained model → ${HORMONES_NOEXT})"
    python ${EXTRACT} \
        --init \
        --size "${SIZE}" --vocab_size "${VOCAB_SIZE}" \
        --out "${HORMONES_NOEXT}"
else
    log "EXTRACTION NOEXT: hormone bank found, skipping → ${HORMONES_NOEXT}"
fi

# =============================================================================
# Step 2 — Seven ablation runs (Stage-2 / full)
# =============================================================================

# ─── Base: continue Stage-1 for 3B more tokens (no hormones) ─────────────────
BASE_DIR="${RUNS_DIR}/${SIZE}_base"
if [ ! -f "${BASE_DIR}/ckpt_final.pt" ]; then
    log "BASE (${STAGE2_TOKENS} more tokens, no mods) → ${BASE_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_DIR}/ckpt_milestone.pt" \
        --out "${BASE_DIR}"
else
    log "BASE: already done → ${BASE_DIR}/ckpt_final.pt"
fi

# ─── IA: continue Stage-1-IA for 3B more tokens (no hormones) ────────────────
# (fallback fn auto-resumes from any in-progress step ckpt; if none exists it falls
#  through to the caller-provided --resume below, i.e. the Stage-1 IA milestone.)
IA_DIR="${RUNS_DIR}/${SIZE}_ia"
if [ ! -f "${IA_DIR}/ckpt_final.pt" ]; then
    log "IA (${STAGE2_TOKENS} more tokens, intracellular attn) → ${IA_DIR}"
    run_ia_with_fallback "${IA_DIR}" \
        --intra_attn \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_IA_DIR}/ckpt_milestone.pt"
else
    log "IA: already done → ${IA_DIR}/ckpt_final.pt"
fi

# ─── HR: extracted hormones, standard scan ────────────────────────────────────
HR_DIR="${RUNS_DIR}/${SIZE}_hr"
if [ ! -f "${HR_DIR}/ckpt_final.pt" ]; then
    log "HR (${STAGE2_TOKENS} tokens, hormone routing) → ${HR_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --hormones --hormone_path "${HORMONES_STD}" \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_DIR}/ckpt_milestone.pt" \
        --out "${HR_DIR}"
else
    log "HR: already done → ${HR_DIR}/ckpt_final.pt"
fi

# ─── Both: IA + extracted hormones ────────────────────────────────────────────
BOTH_DIR="${RUNS_DIR}/${SIZE}_both"
if [ ! -f "${BOTH_DIR}/ckpt_final.pt" ]; then
    log "BOTH (${STAGE2_TOKENS} tokens, IA + hormone routing) → ${BOTH_DIR}"
    run_ia_with_fallback "${BOTH_DIR}" \
        --intra_attn --hormones --hormone_path "${HORMONES_IA}" \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_IA_DIR}/ckpt_milestone.pt"
else
    log "BOTH: already done → ${BOTH_DIR}/ckpt_final.pt"
fi

# ─── HR-rand: random unit-norm hormone vectors ────────────────────────────────
HRRAND_DIR="${RUNS_DIR}/${SIZE}_hr_rand"
if [ ! -f "${HRRAND_DIR}/ckpt_final.pt" ]; then
    log "HR-RAND (${STAGE2_TOKENS} tokens, random vectors) → ${HRRAND_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --hormones --rand_hormones \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_DIR}/ckpt_milestone.pt" \
        --out "${HRRAND_DIR}"
else
    log "HR-RAND: already done → ${HRRAND_DIR}/ckpt_final.pt"
fi

# ─── HR-noext: vectors extracted from untrained model ────────────────────────
HRNOEXT_DIR="${RUNS_DIR}/${SIZE}_hr_noext"
if [ ! -f "${HRNOEXT_DIR}/ckpt_final.pt" ]; then
    log "HR-NOEXT (${TOTAL_TOKENS} tokens from scratch, step-0 vectors) → ${HRNOEXT_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --hormones --hormone_path "${HORMONES_NOEXT}" \
        --tokens "${TOTAL_TOKENS}" \
        --out "${HRNOEXT_DIR}"
else
    log "HR-NOEXT: already done → ${HRNOEXT_DIR}/ckpt_final.pt"
fi

# ─── HR-fixedgate: extracted vectors, gate frozen at 1.0 ─────────────────────
HRFG_DIR="${RUNS_DIR}/${SIZE}_hr_fixedgate"
if [ ! -f "${HRFG_DIR}/ckpt_final.pt" ]; then
    log "HR-FIXEDGATE (${STAGE2_TOKENS} tokens, gate=1.0 frozen) → ${HRFG_DIR}"
    ${LAUNCH} ${TRAIN} ${COMMON} \
        --hormones --hormone_path "${HORMONES_STD}" --fixed_gate \
        --tokens "${STAGE2_TOKENS}" \
        --resume "${STAGE1_DIR}/ckpt_milestone.pt" \
        --out "${HRFG_DIR}"
else
    log "HR-FIXEDGATE: already done → ${HRFG_DIR}/ckpt_final.pt"
fi

# =============================================================================
# Done
# =============================================================================
log "ALL DONE"
echo "Results in ${RUNS_DIR}/${SIZE}_*/"
echo ""
echo "Runs:"
for d in "${RUNS_DIR}/${SIZE}_base" "${RUNS_DIR}/${SIZE}_ia" \
          "${RUNS_DIR}/${SIZE}_hr" "${RUNS_DIR}/${SIZE}_both" \
          "${RUNS_DIR}/${SIZE}_hr_rand" "${RUNS_DIR}/${SIZE}_hr_noext" \
          "${RUNS_DIR}/${SIZE}_hr_fixedgate"; do
    if [ -f "${d}/ckpt_final.pt" ]; then
        echo "  ✓  ${d}"
    else
        echo "  ✗  ${d}  (MISSING)"
    fi
done
