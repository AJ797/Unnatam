# Official Comment: Response to Reviewers

We thank both reviewers for their careful readings and specific, actionable feedback. The revised manuscript addresses every requested change; below we summarise per-reviewer with pointers into the revised PDF. A cloud training run for one remaining ablation (Reviewer D15v's `HR-noext-cont` isolation experiment) is queued and will land as an additional update within 2–5 days; the current revision includes all other changes in full.

---

## Response to Reviewer D15v

Thank you for the positive framing and for identifying the three most load-bearing weaknesses. Each is addressed below.

### 1. HR-NOEXT confounds three variables; the "training schedule is the loss-relevant factor" claim needs isolation.

You are correct — as originally stated, the claim outran the evidence. We took two steps.

**Textual hedging (in the current revision).** The abstract, conclusion, and Section 6.3 headline paragraph now consistently phrase this as *"the only axis along which we found the mechanism could measurably damage the loss is a change in training schedule (with a caveat that we do not fully isolate schedule from the concurrent change in bank content; see Section 6.5)."* Section 7.3 Limitations now includes an explicit "HR-noext confounds three variables" paragraph acknowledging that our schedule attribution is the most plausible reading of the data we have, not a definitively isolated causal claim.

**Isolation experiment (queued; landing in 2–5 days).** We designed a clean isolation ablation — `HR-noext-cont` — which uses the same random-init-extracted hormone bank as `HR-NOEXT` but introduces it as a staged Stage-2 continuation from the standard Stage-1 checkpoint (matching the staged protocol of the other hormone variants). The training script is included in the released code (`scripts/run_stage1_plus_noext_cont.sh`), and Section 6.5 documents the experiment with a three-branch interpretation guide (schedule confirmed / bank content matters / both contribute) that we will populate with the numerical result. The single-session overnight run takes ~22 hours on 8× RTX 4090 and is queued; we will post the result as a further comment within 2–5 days and finalise Section 6.5 in the manuscript accordingly.

### 2. Downgrade Intracellular Attention from a primary architectural contribution to preliminary future work.

Done. In the current revision, IA:
- Is removed from the Contributions bullet list in Section 1.
- Is removed from the abstract's main body (mentioned only in one closing sentence pointing to the appendix as future work).
- No longer has a dedicated main-text section (the previous Section 4 is deleted).
- Is fully consolidated into Appendix C, which opens with an explicit *"We treat this section as future work rather than a claim of the paper: full-budget evaluation is not tractable within our compute budget without a fused CUDA kernel, and the empirical evidence we have is not sufficient to draw conclusions about IA's effect on the loss at scale."*
- The preliminary 12M-token training signal previously in Section 6.5 is moved into Appendix C.3 with the same framing.

### 3. Report statistical uncertainty for downstream benchmark results.

Done. The caption of Table 2 (Main Results) now reports the binomial 95% CI at each benchmark's sample size (HellaSwag ±0.88 pp, LAMBADA ±1.10, ARC-E ±2.00, ARC-C ±2.41, PIQA ±2.24) and notes explicitly that the variant-to-variant gaps in the upper block (max 0.30 pp on ARC-E) are within these bounds while HR-NOEXT's LAMBADA and ARC-E drops are outside them. Figure 3 (zero-shot grouped bars) has been regenerated with binomial 95% CI error bars on every bar. Section 6.3 also now explicitly notes that independent-seed variance was not measured and that the binomial CI is only the eval-set sampling floor — an honest scoping statement, consistent with Reviewer tkcT's request #3.

---

## Response to Reviewer tkcT

Thank you for the detailed critique. We agree with each of the five requested changes; four are addressed in the current revision and the fifth (the seed-variance point) is discussed as a scoping limitation.

### 1. No null baseline for "specialisation"; the L1 numbers could be a metric artifact.

Addressed with a permutation-null test on the trained HR-forced checkpoint, now reported in Section 7.3.

**Method.** We pool the per-position routing distributions at layer 11 across all six genre passages (371 positions total, seven-simplex distributions). We then permute the position→genre assignment 1000 times, computing the max pairwise L1 across the six pseudo-genres each time; the empirical distribution of these values is the null.

**Result.** Over N = 1000 permutations:
- Null median L1 = **0.41**
- Null 95th percentile = **0.56**
- Null 99th percentile = **0.62**
- Observed L1 (HR-forced, layer 11) = **1.17**
- Empirical *p* = **0** (0/1000 permutations produced L1 ≥ observed)

The observed value is ~1.9× the 99th percentile of the null. Because the null distribution depends on the metric and the pool size rather than on the model's trained weights, the same bound applies to the other variants; the observed L1 values of 0.96–1.56 across HR / HR-rand / HR-fixedgate / HR-noext (Figure 4) are all well above the null 99th percentile. The specialisation is not a metric artifact.

The permutation-test script is included in the released code (`scripts/permutation_null_l1.py`).

### 2. "Hormone" implies behavioural steering that is never tested.

Addressed with an inference-time activation-addition steering test on a pretrained Mamba-1.4B model, now reported in Section 7.3 and Appendix E.

We applied our document-style contrast-pair extraction protocol (Section 5.2) to `state-spaces/mamba-1.4b-hf`, extracting seven hormone directions from the residual stream at layers 20–27. At inference we added α · v_i to the residual at those same layers at every token position, generated continuations of five shared prompts under baseline and all seven directions at α ∈ {0.5, 1.5}, and compared. The shifts are unambiguous and semantically aligned with the hormone labels; a selection of examples is in Appendix E. Concretely:

- **Forced ADR (urgency)** at α = 1.5 converts neutral prompts into alarm cascades and countdown sequences (e.g., a prompt about a dog produces *"'He's going to the gym!' 'Don't let him get away!' 'He's going down!' 'Go, go, go!' 'Get him!'"* and an article prompt produces *"the missile.' 'They're going to eject!' … 'Nine, eight, seven, six, five, four, three, two, one!'"*).
- **Forced NRA (persistence)** produces countdown-and-imperative structures (e.g., *"Thirteen, twelve, eleven, ten, nine, eight, seven, six, five, four, three, two, one, zero, now!"* and *"'Five... four... three... two... one... **now!**' '**RUN!**' '**RUN!**' '**RUN!**'"*).
- **Forced OXY (warmth)** produces embracing intimate scenes (*"she stroked its head. 'I'm so happy,' she said. 'Tell me,' she said. 'Tell me.' I'm sitting beside you on your bed, and we're holding each other…"*).
- **Forced SRO (equanimity)** produces contemplative, temporally-extended narratives (*"Many years passed and you did what you had to do. The figures were arranged in order… The people who lived in the valley grew old, and the children grew old…"*).
- **Forced SELF (self-coherence)** produces first-person identity statements (*"I am not a winner. I am not alone. I am not always right. I am not courageous. I am not afraid. I am not a gambler."*).

We take this as evidence that the extraction protocol identifies behaviourally meaningful axes; the "hormone" naming labels an axis that reliably produces the intended behavioural shift when applied at magnitudes above the trained 125M model's homeostatic envelope. We also add an explicit caveat that our 125M trained routing does not itself perform this steering — its effective injection is bounded (Section 6.4) — and that the appendix demonstrates the extraction protocol on a larger model to isolate the "does the naming correspond to a real axis" question from the "does our 125M routing steer" question. The extraction and steering scripts are in the released code (`scripts/extract_hormones_mamba14b.py`, `scripts/steering_test_mamba14b.py`).

### 3. "Noise" claims aren't backed by multiple seeds.

Agreed. We do not have multi-seed reruns, and the "0.30 pp is noise" phrasing was too strong. We took two steps:

- Section 6.3 now explicitly states *"Downstream zero-shot accuracies vary within the binomial 95% CI at each benchmark's sample size (largest variant-to-variant gap 0.30 pp on ARC-E, vs a ±2.0 pp CI). We do not have independent-seed variance, so we cannot rule out that a fraction of these gaps would remain across seeds; the binomial CI is the sampling-noise floor from the eval sets alone."*
- Section 6.4 similarly rephrases the HR / HR-forced comparison at the matched step as *"a 0.02 difference we treat as within the training-noise floor while explicitly noting that we have not measured that floor from independent seeds."*
- Section 7.3 (envelope definition) now defines the envelope width using the lockstep-convergence upper bound (max step-wise divergence < 10⁻³ in loss across the four properly-initialised variants over the entire Stage-2 trajectory) rather than the ambiguous term "statistical noise."

The lockstep-convergence pattern remains stronger evidence than a per-seed CI would provide (Section 7.4 Limitations), but the framing no longer overclaims.

### 4. Add a limiting scope statement to the title and abstract.

Done. The title now reads *"When Structure Doesn't Matter: Loss-Invariant Routing Specialisation in a 125M-Parameter Hybrid Language Model."* The abstract closes with *"All results are at a single scale (125M parameters, 2B tokens, one architecture family, single seed per variant); we hedge accordingly."* The conclusion carries the same scoping statement.

### 5. Either scale up the IA evaluation or move it fully to "Future Work."

Moved fully to future work. See our response to Reviewer D15v #2 for the specifics (IA removed from Contributions, main text, and abstract main body; consolidated in Appendix C which opens with an explicit "we treat this section as future work" framing).

---

## Summary of what's in this revision vs still to come

**In the revised manuscript now (available at [Overleaf link / attached PDF]):**
- ✅ Null-baseline permutation test with numbers (§7.3, tkcT #1)
- ✅ Mamba-1.4B activation-addition steering demonstration (§7.3, Appendix E, tkcT #2)
- ✅ Binomial 95% CIs on Table 2 caption + Figure 3 error bars (D15v #3, tkcT #3)
- ✅ Multi-seed hedging throughout (tkcT #3)
- ✅ Title and abstract scope statement (tkcT #4)
- ✅ IA fully in Appendix C as future work (D15v #2, tkcT #5)
- ✅ Schedule-causality claim hedged in abstract, conclusion, §6.3 (D15v #1, tkcT open-ended)
- ✅ HR-noext confound explicitly acknowledged in Limitations (D15v #1)
- ✅ Formal definition of homeostatic envelope (D15v presentation-clarity)
- ✅ Figure 10 (envelope diagram) added (D15v presentation-clarity)

**Landing in the next 2–5 days:**
- 🕒 HR-noext-cont isolation ablation numbers (D15v #1 strong version). This is a single overnight training run (~22 hours) on 8× RTX 4090; the script is queued and included in the release. Section 6.5 currently holds a three-branch interpretation guide that we will replace with the concrete result and one selected branch as soon as the run completes.

We thank both reviewers again for the depth of engagement. The revision is meaningfully stronger for it.
