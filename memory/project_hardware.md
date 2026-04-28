---
name: Hardware constraints for Unnatam training
description: Current and planned hardware available for Unnatam training and inference
type: project
---

**Current (as of 2026-04-27):** RTX 4050 with 6GB VRAM. Realistically supports training models up to ~3B with bf16 + gradient checkpointing + small batches. Cannot train 8B from scratch on this hardware.

**Planned:** 4090×2 (24GB each, 48GB total). Will enable larger training runs and inference on Llama-Scout-class models.

**Why:** This shapes the scaling experiment for the paper — first set of training runs is 1.7B → 3B on the 4050; bigger runs wait for the 4090 rig.

**How to apply:** Default to bf16 mixed precision, gradient checkpointing, and very small per-device batch sizes. Use vast.ai for inference if user gets impatient. Don't propose 8B-from-scratch training plans on current hardware.
