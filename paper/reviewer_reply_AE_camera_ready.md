# Camera-ready response to the Action Editor (yN5Z)

Thank you for the accept-with-minor-revision decision and for the precise, well-scoped list of changes. We have made all three required revisions and also carried out the encouraged inference-time intervention. Below we address each item; section references are to the revised manuscript.

---

## Required revision 1 — Rewrite abstract, contributions, and conclusion (and reconsider the title) so the central claim is "no detectable difference in validation loss under the tested conditions," and discuss explicitly that the small perturbation magnitude may account for much of this.

Done throughout.

- **Title.** Changed from *"When Structure Doesn't Matter: Loss-Invariant Routing Specialisation…"* to **"When Routing Doesn't Move the Loss: Genre-Associated Routing Specialisation in a 125M-Parameter Hybrid Language Model."** The new title scopes the claim to the loss objective and drops the "doesn't matter" framing, which overstated functional irrelevance.
- **Abstract.** Now leads with *"under the tested conditions we detect no difference in validation loss"* and states explicitly that because the model drives the effective injection to $\|\Delta h\|_2/\|h\|_2 < 0.2\%$ of the residual norm, *"the absence of a detectable loss difference may largely reflect how little the channel ultimately contributes."* We removed the claims that the channel is functionally irrelevant, that predictions are unchanged, and that a capacity bottleneck is ruled out.
- **Contributions (§1).** The primary empirical contribution now reads "no detectable difference in validation loss across routing-channel variants that differ markedly in specialisation," and explicitly says that because the forced variant shrinks the channel ($\|\Delta h\|/\|h\| < 0.2\%$), *"we do not claim to rule out a capacity bottleneck."*
- **Conclusion (§8).** Reworded to match, including an explicit statement that the loss-invariance may reflect how little the channel contributes.

You noted that the model drives the gates down even in the forced variant, so the loss-invariance may largely reflect how little the channel contributes. We agree, and this is now the framing we use in all four places.

## Required revision 2 — Present the homeostatic envelope as a description of the range probed, not an identified mechanism, and remove wording that treats unmeasured seed variance as a noise floor.

Done.

- The envelope is now defined (§7.2) as *"the range of effective injection magnitudes we have empirically probed without measuring a loss difference — not as an identified mechanism or a boundary we have localised."*
- We removed the "training-noise floor" language. The definition now states explicitly that the $<10^{-3}$ lockstep-divergence threshold is **not** a seed-level noise estimate: *"we did not measure independent-seed variance, so we cannot state how much of a validation-loss gap would survive across seeds,"* and that the lockstep agreement of four correlated continuations is weaker evidence than independent seeds would be. The "Scale and seeds" limitation (§7.3) says the same and defers a multi-seed noise floor to future work.
- The three sentences that previously claimed the forced variant "rules out the capacity-bottlenecked reading" (abstract, §1, §7.2, §8) are gone, replaced by the honest statement that a tiny loss-invariant perturbation is exactly what the capacity-bottlenecked reading predicts.

## Required revision 3 — Report the routing controls per variant and per layer rather than for HR-FORCED only (inference only), and either add a same-norm random-direction control to the Appendix C steering example or label it illustrative.

Both done.

- **Per-variant, per-layer controls (§7.2, Table `tab:per-layer`).** We now report the max-pairwise-$L_1$ specialisation at **both** hormone-bearing attention layers (5 and 11) for every variant. For the two checkpoints we retain (\textsc{HR-forced}, \textsc{HR-noext-cont}) we ran the full multi-document, document-level-null control at each layer directly (`scripts/routing_controls.py --layer {5,11}`). For \textsc{HR} and \textsc{HR-noext}, whose original converged checkpoints were archived, the original inference captures recorded the router distribution both averaged over the two attention layers and at layer 11; since there are exactly two hormone-bearing layers, the layer-5 distribution is their exact difference, $l_5 = 2\bar l - l_{11}$ (each reconstructed genre row is a valid probability distribution — we validated non-negativity and unit sum; `scripts/reconstruct_layer5_routing.py`). The consistent finding is that specialisation tracks gate engagement: strong at the engaged upper layer ($L_1 = 0.96$–$1.56$) and near-uniform at the lower layer ($L_1 \approx 0.32$) wherever the lower gate is closed.
  - We are transparent about one cell we cannot fill: \textsc{HR-fixedgate}'s lower gate is frozen *open* ($g_5 = 1.0$), so we expect its layer-5 routing to be specialised as well, but its converged checkpoint was not retained. We report this as a prediction and can supply the number with a same-configuration re-run of that single variant if you consider it necessary (we have retained the Stage-1 checkpoint, so this is one 1B-token Stage-2 continuation).
- **Appendix C steering.** Now explicitly **labelled illustrative**: the setup paragraph states it is "illustrative, not a controlled experiment," that we do not compare against a matched-norm random-direction control, and that such a control (plus classifier-scored shift magnitude) would be needed to establish the claim quantitatively. The §7 reference to it was softened to match.

## Encouraged — Inference-time intervention (uniform/shuffled routing or zero gate) reporting perplexity and the KL divergence between predictive distributions.

Done; we consider this the most informative single result and have added it to §7.2 (`scripts/routing_intervention_kl.py`). On \textsc{HR-forced} — the *largest*-injection trained variant — replacing the learned routing with each intervention changes the predictive distribution negligibly:

| Intervention | $\Delta$PPL | mean KL(real‖intervened) | top-1 flip rate |
|---|---|---|---|
| uniform routing | $+0.003$ | $4.7\times10^{-6}$ nats | $0.12\%$ |
| shuffled routing | $+0.003$ | $7.6\times10^{-6}$ nats | $0.16\%$ |
| zero gate (channel off) | $+0.0001$ | $1.7\times10^{-5}$ nats | $0.28\%$ |

Even removing the routing channel entirely (zero gate) moves the next-token distribution by $\approx 10^{-5}$ nats and flips the argmax on $<0.3\%$ of positions. This is direct evidence that, under the tested conditions, the model's predictions barely depend on the routing structure — which supports exactly the scoped reading you asked us to adopt: the loss-invariance largely reflects how little the channel contributes. Because the unforced variants drive their gates even lower than \textsc{HR-forced}, their predictions depend on the routing even less, so these numbers bound the family. The second retained checkpoint, \textsc{HR-noext-cont} (a normal-gate variant), confirms this with an even smaller effect: uniform, shuffled, and zero-gate interventions all give $\mathrm{KL} \le 6.5\times10^{-7}$ nats and top-1 flip rates $\le 0.07\%$.

---

We believe these changes bring the manuscript's claims in line with the single-seed, one-scale evidence, as you asked. Thank you again for the constructive and precise review.
