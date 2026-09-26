# Follow-up Official Comment: HR-noext-cont isolation ablation completed

We committed in our previous comment to run the `HR-noext-cont` isolation ablation (Reviewer D15v's requested change #1, strong version) within 2–5 days. The training run has now completed. This comment reports the numerical result, the resulting decomposition of the `HR-noext` loss damage, and the manuscript sections revised accordingly. A revised PDF has been uploaded.

## The HR-noext-cont result

`HR-noext-cont` uses the `HR-noext` hormone bank (random-initialization-extracted) but introduces it as a staged Stage-2 continuation from a freshly-retrained Stage-1 checkpoint — matching the staged protocol used by all other hormone variants. It isolates the "training-schedule" contribution from the "bank-content" contribution to `HR-noext`'s loss damage.

**Final numbers** (Table 2, revised):

| Variant | Val PPL | HellaSwag | LAMBADA | ARC-E | ARC-C | PIQA |
|---|---|---|---|---|---|---|
| Base | 26.13 | 28.14 | 20.47 | 43.69 | 23.38 | 61.15 |
| HR-noext-cont *(new)* | **26.56** | 28.34 | 19.12 | 41.71 | 23.55 | 61.26 |
| HR-noext | 28.20 | 27.59 | 18.26 | 40.82 | 23.21 | 61.26 |

## Decomposition of the HR-noext damage

Because `HR-noext` and `HR-noext-cont` share the same bank content and differ only in the training schedule, their gap decomposes the total damage cleanly:

- Total `HR-noext` damage vs Base: **+2.07 PPL**
- Schedule-only contribution (`HR-noext` − `HR-noext-cont`): **+1.64 PPL ≈ 79%**
- Bank-content-only contribution in the staged regime (`HR-noext-cont` − Base): **+0.43 PPL ≈ 21%**

The training-schedule choice is the dominant loss-relevant factor by roughly 4:1, but an uninformative bank contributes a small measurable residual even in the staged regime. Downstream: HR-noext-cont's LAMBADA and ARC-E drop are ~60% and ~70% smaller than HR-noext's, respectively, consistent with the val PPL decomposition.

The finding is intermediate between the two extreme readings we had outlined in the previous comment: schedule is the dominant driver but bank content is not fully neutral.

## Manuscript sections updated

- **Section 6.5** (Isolation ablation) — replaced the TBD block with the numerical result, the decomposition, and the revised conclusion.
- **Section 7.2** (Discussion) — updated the "loss-relevant axis" statement to describe schedule as the dominant (not sole) factor and cite the 79/21% split.
- **Section 7.3** (Limitations) — HR-noext-confound paragraph now reflects that the isolation experiment was run and decomposes the damage; notes the remaining open question (whether an extracted-but-benign bank trained from-scratch would produce the same $+1.64$ PPL schedule effect).
- **Abstract and Conclusion** — schedule-causality wording now cites the 79/21% decomposition explicitly rather than an unqualified schedule claim.
- **Table 2 caption and Figure 4 (envelope diagram)** — updated to include the HR-noext-cont point and its interpretation.

The envelope diagram (Figure 4) in particular visualises the isolation cleanly: `HR-noext-cont` and `HR-noext` sit at nearly the same effective injection magnitude $\mu$ (0.30 vs 0.36) but at very different val PPL (26.56 vs 28.20). The vertical gap between the two same-$\mu$ points is the schedule-only contribution; the small residual of HR-noext-cont above the envelope band is the bank-content contribution.

The `run_stage1_plus_noext_cont.sh` orchestration script used for this run is in the released code.

Thank you again for the specific and actionable ask. The paper's causal claim is now supported by a decomposition rather than a single-observation inference.
