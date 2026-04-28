---
name: Unnatam architecture decisions (v1)
description: Locked-in design choices for the first version of the Unnatam architecture
type: project
---

Decisions agreed during initial design discussion (2026-04-27):

1. **Backbone:** Hybrid SSM (Mamba-style) + MQA Attention, fixed 5:1 ratio (5 SSM blocks per 1 Attention block).
2. **Attention:** MQA (multi-query, single KV head). FlashAttention used as kernel implementation when available — not a separate architectural choice.
3. **Hormone injection points:** After every Attention block only (not every layer — avoids hallucination, keeps story clean).
4. **Hormone vectors:** Fixed direction, learned router + per-hormone magnitude. Vectors loaded from disk as frozen parameters; the router (Linear → Softmax over hormone dictionary) and scalar magnitudes are trained end-to-end.
5. **Hormone extraction strategy (locked, 2026-04-28):** Stage-1 pretrain Unnatam without hormones (gate=0 makes routing a no-op). At ~1B tokens, run `scripts/extract_hormones.py` to produce a `(7, d_model)` `.npy` from **document-style** contrast pairs (NOT instruction-style — the base model isn't instruction-tuned). Extraction = mean residual stream output across all `AttnLayer` injection points, mean across token positions, difference-of-means across pairs, L2-normalize per hormone. Continue training with hormones loaded; gate is now free to grow. **Re-extract at milestones** (1B / 5B / 10B tokens) so hormone vectors track the model's evolving representation space — this becomes part of the paper's "tracking the affective subspace" narrative. Contrast definitions in `unnatam/hormones/definitions.py` (6 pairs × 7 hormones, varied genre).
6. **SSM kernel:** WSL + `mamba-ssm` is the production path (4050 has CUDA passthrough via WSL2). The pure-PyTorch reference scan stays in `unnatam/model/ssm.py` as auto-fallback for Windows host / CPU smoke tests / numerical cross-check of the kernel. Selection is automatic in `MambaBlock.forward` based on `mamba_ssm` import availability + tensor device.
7. **Nested MoE:** Pulled from v1 scope. Likely to underperform on small models (<10B per expert); revisit when scaling up.
8. **Tooling:** Python. Pure-PyTorch implementations preferred for portability (Windows host) and debuggability over kernel-bound libraries; can swap in optimized kernels later.
9. **Tokenizer:** GPT-2 BPE (50k, no auth) by default — see `scripts/prepare_data.py`. User can override to Mistral/Llama via `--tokenizer`. Model `vocab_size` should be set to match.
10. **Dev / training split:** Develop and run smoke tests on Windows host. Run pretokenization, training, and extraction in WSL where `mamba-ssm` and `flash-attn` are installable.

**Why:** These choices prioritize clean ablations for the paper, debuggability on the 4050, and a single coherent novelty story (trained hormone routing) without scope creep.

**How to apply:** Don't add nested MoE, learned SSM/Attn mixing gates, or per-layer hormone injection without explicit user buy-in — those were deliberately deferred or rejected.
