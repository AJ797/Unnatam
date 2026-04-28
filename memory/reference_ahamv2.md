---
name: AhamV2 reference repo
description: Pointer to sibling AhamV2 codebase used as reference for Unnatam's hormone mechanism
type: reference
---

AhamV2 lives at `E:\empire\AhamV2\aham_v2`. It is the user's prior project and the conceptual ancestor of Unnatam's hormone routing.

Key files to consult when designing the equivalent Unnatam mechanism:
- `hormones/definitions.py` — 7 hormone contrast pairs (ADR, CDO, LCO, NRA, OXY, SRO, SELF)
- `hormones/vector_compute.py` — repeng-based extraction from Llama-3.2-3B
- `hormones/vector_store.py` — on-disk storage of `.npy` vectors
- `latent/injector.py` — forward-hook residual injection (AhamV2-style; Unnatam will replace this with a native architectural component)
- `latent/steerer.py` — dual-path (ToM + hormone) shift composition
- `latent/hormone_projector.py` — gated MLP that maps 7-dim hormone state → hidden dim
- `benchmarks/` and `paper/experiment_runner.py` — ablation/eval framework worth mirroring for Unnatam's paper

**Key conceptual difference:** AhamV2 attaches steering to a frozen 70B model via forward hooks; vectors are detached and gradients do not flow through them. Unnatam treats the router and magnitudes as trained parameters with full gradient flow.
