# Response to Reviewer f2c6

Thank you for the detailed review. Every point produced a real improvement. The four requested changes and six additional comments are addressed in the revised manuscript; details on each are in `reviewer_replies_f2c6_full.md` (code supplement). Below are the six substantive critiques.

We would gently note that `HR-noext-cont` (isolation ablation) was posted as a follow-up comment shortly before your review; it partially addresses "cannot separate schedule from bank." The descriptive-vs-causal distinction has been sharpened throughout.

**1. Multi-doc + untrained-router controls.** New `scripts/routing_controls.py`, N=20 held-out docs/genre.
- HR-forced observed $L_1 = 1.16$ (vs. single-passage $1.17$ — not driven by passage choice).
- **Document-level** permutation null (N=2000 shuffles of doc→genre): median $0.30$, 99%ile $0.47$. Observed $\approx 2.5\times$ the 99%ile; empirical $p=0$.
- **Untrained-router baseline** (fresh init, same corpus): $L_1 = 0.076$. Trained is $\approx 15\times$ larger — a learned amplification, not just a projection of input differences.

**2. Numerical inconsistencies + null variant-specificity.**
- Document-level permutation added in §7.2, alongside the token-level test.
- Null depends on the checkpoint's router-output geometry; reported as evidence-of-life on HR-forced, not a universal bound.
- "Chance below 0.1" was wrong — apologies. Abstract now leads with the document-level numbers ($1.16$ vs. $0.47$, $\approx 15\times$ untrained).
- HR $L_1 = 0.60$ vs. $0.96$ was the same variant under two conventions (layer-averaged vs. layer-11-only). Manuscript now **standardises on layer-11** ($0.96$–$1.56$ across five main variants) and states why the lower layer is nearly uniform when its gate is closed.

**3. Single-seed loss-equivalence claim.** Agreed; language hedged.
- §6.3: downstream accuracies within the binomial 95% CI at each benchmark's $n$; independent-seed variance not measured, so binomial CI is the sampling-noise floor from eval sets alone.
- §7.2: envelope now uses explicit "$< 10^{-3}$ in loss across the four properly-initialised variants" instead of an unqualified "statistical noise" term.
- §7.3 Limitations adds "Scale and seeds" paragraph noting multi-seed variance would be needed to firmly establish a training-noise floor. We do not run independent seeds this revision because of compute cost (~2–3 additional 22-hour multi-GPU runs); stated explicitly.

**4. Envelope not operationally established; $\mu$ omits per-token weights + residual scale; $\mu \le 0.5$ contradicts HR-FIXEDGATE.**
- **$\mu \le 0.5$ contradiction fixed.** Envelope now contains at least $\mu \in [0, 1.0]$ (upper end set by HR-FIXEDGATE). Apologies.
- **Actual perturbation measured.** Controls script reports $\|\Delta h\|_2/\|h\|_2$ per attention layer per document. HR-forced (largest-perturbation trained variant): $0.15\%$ layer 5, $0.07\%$ layer 11. Under $2\times$-amplified forced engagement the routing channel is well under $1\%$ of residual-stream norm. Abstract, §7.2, Figure 4 caption now cite these.
- **One magnitude can't identify the envelope.** Agreed; §7.2 now calls it *"descriptive of the region we have empirically probed, not a fully-identified boundary."*
- **Capacity bottleneck not ruled out.** §7.3 explicitly: amplifying a fixed 7-D subspace does not expand information capacity; the capacity-bottlenecked interpretation could re-emerge at larger $d_{\text{model}}$ or longer training.

**5. 79/21% split is descriptive, not causal.** Agreed. §7.3 already acknowledged this; framing now tightened throughout.
- Abstract: *"descriptive comparison between two observed conditions rather than a fully identified crossed causal decomposition (a from-scratch-with-extracted-good-bank arm would be required; we leave it to future work)."*
- §6.5 "Decomposition" heading → "Descriptive comparison"; each quantity a *delta* between observed conditions.
- Conclusion, §7.2, §7.3, Figure 4 caption: schedule described as "the axis along which we observe measurable damage"; the split as "schedule delta at fixed bank" ($+1.64$ PPL) and "bank delta at fixed schedule" ($+0.43$ PPL). Full crossed 2×2 is on the roadmap.

**6. Missing citations (ReFT / CS-ReFT / Guiding Giants / RISER).** Added. §2 now has a dedicated "Routed representation-intervention methods" paragraph citing all four, with explicit positioning relative to RISER (closest neighbour): different setting (continued pretraining vs. supervised finetuning), different bank source (contrast-pair extraction vs. external definition), different framing (routing-loss decoupling as a *finding* vs. router-as-intervention as a *design aim*). The RISER pointer was especially useful — thank you.

Thank you again — this is a much better paper for the depth of the review.
