# Response to Reviewer f2c6

Thank you for the careful, technically detailed review. Every point is addressable and every one has produced a real improvement in the manuscript. We have (a) fixed the numerical inconsistencies you flagged, (b) softened the causal framing of the schedule/bank comparison, (c) added the routing controls you requested (multi-document + document-level permutation + untrained-router baseline + actual residual perturbation magnitude), (d) added the related-work citations you named, and (e) addressed each of the six additional comments. Below we go through each requested change and each critique.

We would gently note that the `HR-noext-cont` isolation ablation was posted as a follow-up official comment shortly before your review; some of the "cannot separate schedule from bank" and "descriptive not causal" concerns are partially addressed there. We have taken the opportunity to sharpen the framing further so the descriptive-vs-causal distinction is unambiguous.

---

## Response to the six substantive critiques

### 1. "Genre analysis uses only one selected passage per genre; needs multi-doc + untrained-router baseline."

Fixed with concrete numbers. We ran a **multi-document routing analysis** and an **untrained-router baseline** using a new script (`scripts/routing_controls.py`) on $N = 20$ held-out documents per genre (news from CNN/DailyMail, fiction from TinyStories, code from The Stack, dialog from PersonaChat, tech from arXiv summarisation, urgent from CNN/DailyMail filtered for emergency keywords).

**Multi-document analysis, HR-forced checkpoint:**

- Observed max pairwise $L_1$ (per-genre means over 20 documents each) $= 1.16$.
- Compare to the single-passage value from the original submission $= 1.17$. The specialisation is **not** driven by our particular passage choice — it holds under a 20$\times$ increase in documents per genre with essentially the same $L_1$.

**Document-level permutation null** ($N = 2000$ shuffles of document $\to$ genre assignment, so documents are the independent unit, not tokens):

- Null median $L_1 = 0.30$, 95th percentile $= 0.41$, 99th percentile $= 0.47$.
- Observed $L_1 = 1.16$ is $\approx 2.5\times$ the null 99th percentile; empirical $p = 0/2000$.
- This is meaningfully tighter than the token-level null we originally reported (99%ile $= 0.62$); the observed value is above both.

**Untrained-router baseline** (freshly-initialised Unnatam, same architecture, same corpus):

- Untrained observed $L_1 = 0.076$, itself slightly above its own document-level null 99%ile ($0.046$; $p = 0$) — because the underlying hybrid transformer's hidden states are already input-dependent, so even a random projection produces a weak genre signal. This is expected and honest.
- The trained HR-forced router's $L_1$ is $\approx 15\times$ larger ($1.16$ vs $0.076$): training amplifies the input-driven baseline by more than an order of magnitude. The observed specialisation is not simply a projection of input-side differences — it is a genuinely learned amplification of them.

Both JSONs are in the released code (`runs/tiny_hr_forced/routing_controls.json`, `runs/untrained/routing_controls.json`).

### 2. "Permutation test treats 371 tokens as independent samples; needs document-level; also HR-FORCED null does not automatically apply to other variants; numerical inconsistencies (chance < 0.1 vs 0.41; HR L1 = 0.60 vs 0.96)."

All fixed.

- **Document-level permutation.** The new controls script permutes at the *document* level (not the token level), with $N = 2000$ shuffles of the per-document mean routing rows. Reported in the revised §7.2 alongside the original token-level test.
- **Null does not automatically apply to every variant.** We agree. The revised §7.2 now says the permutation null is reported on `HR-forced` and notes that the null distribution depends on the checkpoint's router-output geometry, so the same numerical bound is not automatically transferable; we report it as evidence-of-life for the metric on the checkpoint we tested rather than as a universal bound.
- **"Chance level below 0.1" was wrong.** We apologise for the sloppy phrasing in the earlier abstract. The abstract now correctly reports: null median $L_1 = 0.41$, 99th percentile $0.62$, observed $L_1 = 1.17$ on HR-forced (empirical $p = 0$; observed $\approx 1.9\times$ the 99th percentile).
- **HR $L_1 = 0.60$ vs $0.96$.** These were the same variant measured with two conventions ($0.60$ averages across both attention layers, $0.96$ is the layer-11-only value). The manuscript now **standardises on layer-11 only** ($0.96$--$1.56$ across the five main variants) and states explicitly that the layer-averaged number is lower because the lower attention layer is nearly uniform when its gate is closed. Range strings and figure captions updated throughout.

### 3. "Loss equivalence claim rests on single seed; correlated Stage-2 continuations don't define a training-noise floor."

We agree that lockstep convergence across correlated Stage-2 continuations from the same Stage-1 checkpoint does not fully substantiate a "training-noise floor" in the multi-seed sense you mean. We have hedged the language:

- §6.3 now says explicitly: *"Downstream zero-shot accuracies vary within the binomial 95% CI at each benchmark's sample size... We do not have independent-seed variance, so we cannot rule out that a fraction of these gaps would remain across seeds; the binomial CI is the sampling-noise floor from the eval sets alone."*
- §6.4 (HR-forced comparison) now says: *"a 0.02 difference we treat as within the training-noise floor while explicitly noting that we have not measured that floor from independent seeds."*
- §7.2 envelope definition now uses the lockstep-convergence bound explicitly ("$< 10^{-3}$ in loss across the four properly-initialised variants") rather than an unqualified "statistical noise" term.
- §7.3 Limitations now includes an explicit "Scale and seeds" paragraph acknowledging single-seed, single-scale, single-architecture as design constraints, and noting that multi-seed variance would be needed to firmly establish a training-noise floor.

We do not run independent seeds in this revision because of compute cost (approximately 2--3 additional 22-hour multi-GPU runs would be required); this is now stated explicitly rather than absorbed into a claim.

### 4. "Homeostatic envelope is not operationally established; $\mu$ omits per-token router weights and residual-stream scale; stated $\mu \leq 0.5$ conflicts with HR-FIXEDGATE at $\mu = 1$; testing one additional magnitude cannot identify the envelope; amplifying a 7-D subspace does not increase capacity."

All valid; the framing has been considerably softened.

- **$\mu$-vs-HR-FIXEDGATE contradiction fixed.** The revised §7.2 states that the envelope contains at least the tested range $\mu \in [0, 1.0]$ (upper end set by HR-FIXEDGATE at $\mu = 1$), not $\mu \leq 0.5$ as previously written. We apologise for the contradiction.
- **$\mu$ omits router weights and residual scale.** We agree. The controls script now measures $\|\Delta h\|_2 / \|h\|_2$ per attention layer per document. For \textsc{HR-forced} (the largest-perturbation trained variant we have), the mean $\|\Delta h\|_2 / \|h\|_2 = 0.15\%$ at layer $5$ and $0.07\%$ at layer $11$. Even under $2\times$-amplified forced engagement the routing channel's actual contribution is well under $1\%$ of the residual-stream norm — a concrete quantification of what the ``bounded homeostatic envelope'' means empirically. The abstract, §7.2, and Figure 10 have been updated to cite these numbers instead of relying on the abstract $\mu$ scalar alone.
- **One-magnitude testing cannot identify the envelope.** Correct. The revised §7.2 now describes the envelope as *"descriptive of the region we have empirically probed, not a fully-identified boundary"* and explicitly states we do not identify its boundary; doing so would require probing multiple $\mu$ values with independent runs.
- **Amplifying a 7-D subspace does not add information capacity — cannot rule out capacity bottleneck.** Correct. §7.3 Limitations now says: *"the capacity-bottlenecked interpretation we partly rule out at this scale could re-emerge in different form at larger $\dmodel$ or longer training; amplifying a fixed 7-dimensional subspace does not by itself expand the information the mechanism can carry."*

### 5. "79/21% split is descriptive rather than causal; needs crossed schedule-by-bank design."

We agree, and we would note that the manuscript already acknowledged this in §7.3 Limitations: *"This does not fully identify the interaction between the two: we do not know, for example, whether an extracted-but-benign bank (like HR's) trained from scratch would produce the same $+1.64$ PPL schedule effect or a smaller one. The stronger causal claim would require that further ablation, which we leave to future work."* We have now sharpened the framing throughout so the descriptive-vs-causal distinction is unambiguous:

- **Abstract:** *"This is a descriptive comparison between two observed conditions rather than a fully identified crossed causal decomposition (a from-scratch-with-extracted-good-bank arm would be required for the latter; we leave it to future work)."*
- **§6.5:** the itemised "Decomposition" heading is now "Descriptive comparison of the tested conditions" and explicitly labels each quantity as a *delta* between observed conditions rather than a causal contribution.
- **Conclusion, §7.2, §7.3, Figure 10 caption:** all updated to describe schedule as "the axis along which we observe measurable damage" rather than "the dominant loss-relevant axis," and to describe the $1.64$/$0.43$ PPL split as "the schedule delta at fixed bank" and "the bank delta at fixed schedule" respectively.

The full crossed 2×2 (adding a from-scratch-with-HR-bank cell) is on our roadmap but requires a further $\sim \$25$ overnight run; we did not want to hold the revision on that.

### 6. "Should cite ReFT, CS-ReFT, Guiding Giants, especially RISER."

Added. The revised §2 Related Work now includes a dedicated paragraph "Routed representation-intervention methods" citing all four, with an explicit positioning of our contribution relative to RISER (closest neighbour): different setting (continued pretraining vs supervised finetuning), different bank source (contrast-pair extraction vs external definition), and different framing (routing-loss decoupling as a finding vs router-as-intervention interface as a mechanism-design aim). Thank you for the pointer to RISER especially — this comparison is genuinely useful for readers locating our contribution.

---

## Response to the four requested changes

1. **"Establish routing structure is learned; document-level + untrained controls."** — Done via the new controls script (§7.2 + released code). See critique 1 above.

2. **"Sufficient evidence for loss equivalence and envelope."** — Language hedged throughout to "no loss difference detected in the reported runs" and "the tested range" rather than "envelope identified"; independent seeds acknowledged as necessary for the stronger claim and deferred to future work. See critique 3.

3. **"Reconsider causal interpretation of 79/21."** — Reframed as descriptive throughout; abstract, §6.5, §7.2, §7.3, conclusion, and Figure 10 caption updated. See critique 5.

4. **"Resolve numerical/visual inconsistencies, clarify optimisation, update related work."** — All numerical inconsistencies fixed (see critique 2, 4). NeurIPS checklist instruction block removed from the submitted PDF. Figure 1 rebuild included with the revised submission (the empty figure in the version you reviewed was a compile artefact of a stale build). Related work extended (critique 6).

---

## Response to the six additional comments

1. **"§5.2 says base model has never seen 'You are a helpful assistant'; verify or soften."** — Softened to *"FineWeb-Edu is dominated by prose passages rather than chat-formatted instructions, so a system prompt of the form 'You are a helpful assistant' is far outside the typical input distribution."* No longer claims the model has literally never seen such text.

2. **"'Greedy multinomial sampling' is contradictory."** — Fixed. Now reads *"multinomial sampling at temperature 0.7 with seed 0."*

3. **"Paper alternates between five main variants, five routing variants, additional forced/continuation."** — Added an explicit naming-convention clarification at the start of §6.2: *"We refer throughout to the same five 'main' hormone-routing variants (Base, HR, HR-rand, HR-fixedgate, HR-noext) plus two extensions reported separately (HR-forced, HR-noext-cont). 'Five variants' always means the main comparison; extensions are named explicitly."*

4. **"'Upper layers carry more decision-relevant signal' is only indirectly supported."** — Reframed as a hypothesis in §7.1: *"A working hypothesis, consistent with residual-stream organisation observed in transformers [Zou et al 2023], is that the upper attention layer carries more decision-relevant signal... Our two attention layers are not sufficient to test this directly; the layer-asymmetric pattern is compatible with several alternative explanations (e.g., gradient-magnitude asymmetries independent of decision-relevance) and we do not attempt to distinguish them here."*

5. **"Please define exactly how activations are averaged (masking, truncation, length matching)."** — Added to §5.2: *"For each contrast and each pair, we tokenise the document with the GPT-2 BPE (truncated at 256 tokens), run a forward pass, and capture the residual-stream activation immediately after each attention block's residual addition. We average over the two attention layers and over all valid token positions (no masking beyond the standard causal mask; no padding; pairs are processed independently). Positive and negative documents are not length-matched — their token counts vary by a factor of $\leq 2$ across pairs."*

6. **"NeurIPS checklist instruction block is still in the submitted PDF."** — Removed. The template instructions block is gone from `checklist.tex`; only the section heading, the questions, the answers, and the guidelines remain.

---

## Summary of manuscript changes for this reviewer's asks

- Numerical inconsistencies fixed: chance-level phrasing (abstract), HR $L_1$ convention standardised to layer-11 ($0.96$–$1.56$), envelope $\mu$ upper bound corrected ($\mu \in [0, 1.0]$)
- Descriptive-vs-causal framing tightened throughout: schedule/bank comparison now labelled as observed-conditions delta, not causal decomposition
- Multi-document (20 docs/genre), document-level-permutation ($N=2000$), and untrained-router controls added: HR-forced observed $L_1 = 1.16$ vs untrained $0.076$ ($\approx 15\times$ smaller) and document-level null 99%ile $0.47$ (§7.2)
- $\|\Delta h\|_2 / \|h\|_2 = 0.15\%$ / $0.07\%$ at attention layers 5 / 11 for HR-forced — concrete quantification of the perturbation magnitude
- Envelope now described as tested-region, not identified-boundary
- Multi-seed variance acknowledged as design constraint in §7.3 Limitations
- Related work extended with ReFT / CS-ReFT / Guiding Giants / RISER
- All six additional comments addressed

Thank you again — this is a much better paper for the depth of the review.
