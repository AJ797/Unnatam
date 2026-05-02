#!/usr/bin/env bash
# Print every IA gate value in the trained checkpoint.
# A non-zero gate after training is the proof that the model chose to use the
# intracellular-attention mechanism — not just had it available.
#
# Usage:
#   bash scripts/inspect_ia_gates.sh
#   CKPT=/workspace/runs/tiny_ia/ckpt_final.pt bash scripts/inspect_ia_gates.sh

CKPT="${CKPT:-/workspace/runs/tiny_ia/ckpt_final.pt}"

source /venv/main/bin/activate 2>/dev/null || true

python3 - <<EOF
import torch
ckpt = torch.load("${CKPT}", map_location="cpu", weights_only=False)
sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

print(f"Inspecting: ${CKPT}")
print(f"Total params in state_dict: {len(sd)}\n")

ia_keys = [k for k in sd.keys() if "intra" in k.lower() or "ia" in k.lower().split(".")]
print(f"Found {len(ia_keys)} IA-related parameters")
print()

# Print scalar / small parameters (gates), then a summary of larger ones (Q,K projections)
print("─── Scalar / gate parameters (these MUST be non-zero post-training) ───")
for k in sorted(ia_keys):
    v = sd[k]
    if v.numel() == 1:
        print(f"  {k:<60s} = {v.item():+.6f}")
    elif v.numel() <= 16:
        vals = ", ".join(f"{x:+.4f}" for x in v.flatten().tolist())
        print(f"  {k:<60s} = [{vals}]")

print()
print("─── Projection norms (sanity: should not be zero) ───")
for k in sorted(ia_keys):
    v = sd[k]
    if v.numel() > 16:
        print(f"  {k:<60s}  shape={tuple(v.shape)}  ‖·‖₂={v.float().norm().item():.4f}")
EOF
