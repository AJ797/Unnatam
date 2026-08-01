# When Structure Doesn't Matter: Loss-Invariant Routing Specialisation in Hybrid Language Models

> A study of emergent routing behaviour in hybrid state-space/attention language models, showing that semantically meaningful routing can arise independently of language modelling performance.

---

## Abstract

This work investigates whether internal architectural structure in language models necessarily contributes to optimisation performance. We introduce **hormone routing**, a token-conditional routing mechanism that injects a gated combination of frozen direction vectors into the residual stream of a hybrid **Mamba + Multi-Query Attention (MQA)** language model.

Across six controlled experimental variants trained under identical conditions, we find that the router consistently develops **semantically coherent routing specialisation** despite having little or no measurable effect on language modelling loss. Surprisingly, the strongest routing specialisation emerges when the gating mechanism is **unable to modulate its own output**.

These results suggest the existence of a **homeostatic envelope**: a regime where substantial internal representational organisation can evolve while remaining effectively invisible to the optimisation objective. Within this envelope, the model absorbs architectural perturbations without affecting predictions. The only intervention that consistently harms performance is *when* the perturbation is introduced during training—not *what* information it contains.

The paper also introduces **intracellular attention**, a parameter-efficient slot-attention mechanism operating inside the state-space scan, presented as a preliminary architectural extension.

---

## Key Contributions

* Introduces **Hormone Routing**, a lightweight routing mechanism for hybrid language models.
* Demonstrates that routing develops **semantic specialisation** regardless of the injected vector structure.
* Shows that routing specialisation is **loss-invariant**, with four variants converging to identical validation perplexity (**26.13 PPL**).
* Identifies a **homeostatic envelope**, where residual perturbations are actively absorbed by the network.
* Shows that **training schedule** is the only experimentally observed factor capable of breaking this invariance.
* Proposes **Intracellular Attention**, a parameter-efficient extension for State Space Models (preliminary).

---

## Model Architecture

The experiments use a **125M-parameter hybrid language model** combining:

* Mamba State Space layers
* Multi-Query Attention (MQA)
* Residual hormone routing after each attention block

The hormone router computes

```
routing weights
        ↓
Softmax Gate
        ↓
Frozen Direction Bank
        ↓
Weighted Combination
        ↓
Residual Injection
```

The injected vectors are **not trainable**; only the routing network learns how to combine them (except in the frozen-gate control).

---

## Experimental Variants

Six routing configurations were evaluated under an identical **2B-token FineWeb-Edu pretraining budget**.

| Variant         | Frozen Direction Vectors                  | Gate                  | Purpose                                 |
| --------------- | ----------------------------------------- | --------------------- | --------------------------------------- |
| Extracted       | From pretrained model                     | Trainable             | Main method                             |
| Random Unit     | Random unit vectors                       | Trainable             | Tests dependence on semantic structure  |
| Random Model    | Extracted from randomly initialised model | Trainable             | Tests dependence on pretrained geometry |
| Frozen Gate     | Extracted vectors                         | Fixed at 1.0          | Removes gate modulation                 |
| Amplified       | Forced non-zero gate + 3× amplification   | Partially constrained | Tests limits of perturbation absorption |
| Early Injection | Router introduced during base pretraining | Trainable             | Tests timing effects                    |

---

## Main Results

### 1. Routing Specialisation Emerges Universally

Every routing variant develops meaningful genre-dependent routing behaviour.

Observed maximum pairwise L1 distances between genre routing distributions:

```
0.6 – 1.6

where

0   = identical routing
2   = completely disjoint routing
```

---

### 2. Specialisation Does Not Predict Performance

Four independently initialised variants converge to essentially identical validation performance.

| Variant      | Validation PPL |
| ------------ | -------------: |
| Extracted    |          26.13 |
| Random Unit  |          26.13 |
| Random Model |          26.13 |
| Frozen Gate  |          26.13 |

The strongest routing specialisation occurs in the **Frozen Gate** variant, despite the gate being incapable of adaptation.

---

### 3. Evidence for a Homeostatic Envelope

To test whether the loss invariance represented a ceiling or an actively maintained regime, an amplified perturbation was introduced.

Despite:

* forced non-zero gate initialisation,
* 3× residual amplification,
* inability to completely suppress injected signals,

the model:

* reduced gate values,
* decreased per-hormone magnitudes,
* maintained validation loss.

The remaining perturbation was approximately **2× larger** than in the unconstrained regime yet still produced no measurable degradation.

This motivates the concept of a **homeostatic envelope**: a region of architectural perturbation that the model consistently absorbs without affecting predictions.

---

### 4. Timing Matters More Than Content

The only intervention that significantly degrades language modelling performance is introducing hormone routing during initial base pretraining.

```
Validation Perplexity Increase

+2.07 PPL
```

Importantly, routing specialisation still develops under this condition.

This suggests that the optimisation objective is sensitive to **when** the perturbation enters training rather than to its semantic content or magnitude.

---

## Intracellular Attention

The paper additionally proposes **Intracellular Attention**, a lightweight slot-attention mechanism embedded inside the Mamba state-space scan.

The design aims to:

* improve parameter efficiency,
* increase representational capacity,
* preserve linear-time inference.

Current results are preliminary, as full evaluation requires a fused CUDA implementation.

---

## Central Finding

The principal conclusion is that **internal representational organisation and optimisation performance can become decoupled**.

Meaningful, semantically structured routing behaviour emerges even when:

* routing directions are random,
* routing geometry is unrelated to pretrained representations,
* gates cannot adapt,
* injected signals are artificially amplified.

Within a broad homeostatic regime, these architectural differences are effectively invisible to the language modelling objective.

-----------

## Paper link

https://zenodo.org/records/21044635
