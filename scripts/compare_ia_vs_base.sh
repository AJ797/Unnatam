#!/usr/bin/env bash
# Print val_loss(step) side-by-side for IA vs Base, for the first N steps of
# Stage 2. Both runs resumed from the same Stage-1 checkpoint, so step counts
# in their respective train.jsonl files refer to the same Stage-2 progress.
#
# Usage:
#   bash scripts/compare_ia_vs_base.sh
#   IA_DIR=/workspace/runs/tiny_ia BASE_DIR=/workspace/runs/tiny_base \
#       bash scripts/compare_ia_vs_base.sh

IA_DIR="${IA_DIR:-/workspace/runs/tiny_ia}"
BASE_DIR="${BASE_DIR:-/workspace/runs/tiny_base}"

source /venv/main/bin/activate 2>/dev/null || true

python3 - <<EOF
import json, os
from collections import OrderedDict

def load_val_losses(path):
    """Return OrderedDict: step -> val_loss (only entries that have a val_loss)."""
    out = OrderedDict()
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in d and d["val_loss"] is not None:
                step = d.get("step") or d.get("optim_step") or d.get("global_step")
                if step is not None:
                    out[int(step)] = float(d["val_loss"])
    return out

ia_log   = "${IA_DIR}/train.jsonl"
base_log = "${BASE_DIR}/train.jsonl"

ia   = load_val_losses(ia_log)
base = load_val_losses(base_log)

if not ia:
    print(f"WARN: no val_loss entries in {ia_log}")
if not base:
    print(f"WARN: no val_loss entries in {base_log}")

# Show every IA val-loss step and the closest Base step (within ±50 steps)
def closest(target, keys):
    if not keys: return None
    return min(keys, key=lambda s: abs(s - target))

print(f"{'step':>8s}  {'IA':>10s}  {'Base':>10s}  {'Δ (IA-Base)':>14s}")
print("─" * 50)
for step, v_ia in ia.items():
    nearest = closest(step, list(base.keys()))
    if nearest is None or abs(nearest - step) > 50:
        print(f"{step:>8d}  {v_ia:>10.4f}  {'—':>10s}  {'—':>14s}")
    else:
        v_base = base[nearest]
        diff = v_ia - v_base
        marker = " ✓" if diff < 0 else " ✗"
        print(f"{step:>8d}  {v_ia:>10.4f}  {v_base:>10.4f}  {diff:>+14.4f}{marker}")

# Crude summary
if ia and base:
    common = [(s, ia[s], base[closest(s, list(base.keys()))])
              for s in ia.keys() if closest(s, list(base.keys())) is not None]
    if common:
        deltas = [v_ia - v_base for _, v_ia, v_base in common]
        n_better = sum(1 for d in deltas if d < 0)
        print()
        print(f"IA beats Base at {n_better}/{len(deltas)} matched eval steps")
        print(f"Mean Δ (IA - Base) = {sum(deltas)/len(deltas):+.4f}  (negative = IA wins)")
EOF
