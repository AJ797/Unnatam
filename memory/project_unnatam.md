---
name: Project Unnatam
description: New LLM architecture being built from scratch as research project, intended to produce a paper
type: project
---

Unnatam is a from-scratch LLM architecture project at E:\empire\Unnatam. It is the user's follow-up to AhamV2 and is intended to produce a paper.

**Why:** The user wants to demonstrate that affective steering ("hormone" injection) can be a first-class trained architectural component rather than post-hoc inference-time steering as in AhamV2. The hybrid SSM/Attention base gives the paper a modern foundation; the trained hormone routing is the novel contribution.

**How to apply:** When working on Unnatam, treat it as research code targeting a paper — clean ablation paths matter, scaling graphs matter (1.7B → 3B for now, up to 8B/Llama-Scout when bigger hardware arrives), and the hormone-routing piece is the headline novelty. The hybrid SSM/Attn is supporting architecture, not the contribution itself.

Sibling project: AhamV2 at E:\empire\AhamV2\aham_v2 — reuse its hormone contrast definitions, vector extraction approach, and benchmarking philosophy. Key difference: AhamV2 uses inference-time forward hooks on a frozen Llama-70B; Unnatam trains the routing end-to-end into the architecture.
