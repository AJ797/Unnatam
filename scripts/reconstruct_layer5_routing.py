"""Per-layer routing L1 for the variants whose checkpoints were not retained.

Background. The camera-ready (Action Editor request) asks for the routing
specialisation controls *per variant and per layer*, not for HR-FORCED only.
For HR-FORCED and HR-noext-cont we compute this directly from the checkpoints
(scripts/routing_controls.py, --layer 5 and --layer 11). For HR, HR-rand,
HR-fixedgate and HR-noext the original converged checkpoints were not retained
after the first submission. However, the inference logs from the original runs
recorded, for HR and HR-noext, BOTH:

  (a) the per-genre router distribution averaged over all hormone-bearing
      layers ("avg"), and
  (b) the per-genre router distribution at layer 11 only ("l11").

The model has exactly two hormone-bearing attention layers (indices 5 and 11),
so avg = (l5 + l11) / 2 and therefore

      l5 = 2 * avg - l11                                (exact, not an estimate)

is fully determined by two measured quantities. This script holds the logged
tables verbatim, reconstructs l5, VALIDATES it (each genre row must be a valid
probability distribution: non-negative and summing to 1), and reports the
max-pairwise-L1 specialisation at each layer. For HR-rand and HR-fixedgate only
the l11 table was logged; we report l11 and, for HR-rand (lower gate closed,
|g_5| = 7e-4), note that its layer-5 routing is unconstrained by gradient and
empirically near-uniform as in the other closed-gate variants. HR-fixedgate has
its lower gate frozen OPEN at 1.0, so its layer-5 routing is expected to be
specialised; that single cell requires the checkpoint and is left as the one
gap (a same-config re-run would fill it if the editor considers it necessary).

Provenance: tables copied from the original-run inference logs
(`archaeology` capture dumps). Reproduce the capture on any retained checkpoint
with scripts/routing_controls.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GENRES = ["news", "fiction", "code", "dialog", "tech", "urgent"]
HORMONES = ["ADR", "CDO", "LCO", "NRA", "OXY", "SRO", "SELF"]

# ── Logged tables (avg over both hormone layers, and layer-11 only) ──────────
HR_AVG = np.array([
    [0.3771, 0.0949, 0.1121, 0.0723, 0.1152, 0.1458, 0.0825],
    [0.2480, 0.3085, 0.0739, 0.0659, 0.0583, 0.1456, 0.0999],
    [0.3053, 0.0706, 0.2279, 0.1071, 0.0918, 0.1587, 0.0387],
    [0.3246, 0.1817, 0.0528, 0.0892, 0.0906, 0.1950, 0.0660],
    [0.3571, 0.2068, 0.1009, 0.1040, 0.0686, 0.1006, 0.0619],
    [0.2286, 0.0962, 0.1138, 0.0982, 0.1859, 0.1518, 0.1255]])
HR_L11 = np.array([
    [0.5622, 0.0482, 0.0554, 0.0252, 0.0980, 0.1889, 0.0221],
    [0.3308, 0.4358, 0.0264, 0.0013, 0.0063, 0.1382, 0.0611],
    [0.4475, 0.0324, 0.2861, 0.0161, 0.0290, 0.1745, 0.0144],
    [0.4956, 0.1940, 0.0056, 0.0129, 0.0366, 0.2489, 0.0063],
    [0.5669, 0.2191, 0.0170, 0.0951, 0.0217, 0.0763, 0.0039],
    [0.3070, 0.0181, 0.0748, 0.0764, 0.2451, 0.1584, 0.1202]])

HRNOEXT_AVG = np.array([
    [0.1551, 0.0962, 0.2257, 0.1492, 0.2025, 0.0958, 0.0754],
    [0.0700, 0.2159, 0.3477, 0.0887, 0.0764, 0.1260, 0.0752],
    [0.1088, 0.0676, 0.2359, 0.0582, 0.1385, 0.0719, 0.3191],
    [0.0751, 0.3757, 0.2300, 0.1045, 0.0676, 0.0843, 0.0628],
    [0.1038, 0.0894, 0.1810, 0.3012, 0.1197, 0.0802, 0.1248],
    [0.0810, 0.0906, 0.2903, 0.0839, 0.1072, 0.1555, 0.1914]])
HRNOEXT_L11 = np.array([
    [0.1635, 0.0971, 0.2644, 0.1775, 0.2021, 0.0716, 0.0238],
    [0.0148, 0.3018, 0.5862, 0.0622, 0.0097, 0.0169, 0.0084],
    [0.0210, 0.0034, 0.3255, 0.0086, 0.1089, 0.0106, 0.5221],
    [0.0145, 0.6106, 0.2939, 0.0703, 0.0042, 0.0040, 0.0025],
    [0.0898, 0.0182, 0.2372, 0.5289, 0.0083, 0.0032, 0.1144],
    [0.0163, 0.0538, 0.4581, 0.0587, 0.0383, 0.1104, 0.2643]])

# Layer-11-only tables (no avg logged → l5 not reconstructible)
HRRAND_L11 = np.array([
    [0.0134, 0.3013, 0.0301, 0.1071, 0.1258, 0.0944, 0.3278],
    [0.1371, 0.5428, 0.0040, 0.1875, 0.0273, 0.0448, 0.0565],
    [0.0079, 0.3366, 0.0138, 0.1377, 0.2788, 0.2180, 0.0073],
    [0.0572, 0.4190, 0.0887, 0.0399, 0.0843, 0.0105, 0.3004],
    [0.0689, 0.0281, 0.0742, 0.2697, 0.4634, 0.0759, 0.0199],
    [0.0817, 0.0985, 0.1067, 0.0619, 0.2751, 0.0148, 0.3612]])
HRFIXEDGATE_L11 = np.array([
    [0.5760, 0.1270, 0.0301, 0.1268, 0.0745, 0.0541, 0.0115],
    [0.5147, 0.0044, 0.0166, 0.0296, 0.1198, 0.2471, 0.0678],
    [0.1701, 0.7816, 0.0080, 0.0165, 0.0013, 0.0181, 0.0045],
    [0.4245, 0.0062, 0.0017, 0.1618, 0.0095, 0.2765, 0.1197],
    [0.6025, 0.1283, 0.0347, 0.0536, 0.0234, 0.1377, 0.0197],
    [0.3316, 0.2809, 0.0027, 0.1321, 0.0439, 0.1379, 0.0708]])

# Post-training per-layer gates (logged)
GATES = {
    "hr":           {5: -0.0002, 11: +0.2314},
    "hr_rand":      {5: +0.0007, 11: -0.1217},
    "hr_fixedgate": {5: 1.0,     11: 1.0},
    "hr_noext":     {5: +0.012,  11: +0.366},  # from fig5_gates (lower gate closed)
}


def max_pairwise_l1(M):
    return float(np.abs(M[:, None, :] - M[None, :, :]).sum(-1).max())


def reconstruct_l5(avg, l11, name, tol=1e-3):
    l5 = 2.0 * avg - l11
    row_sums = l5.sum(axis=1)
    if np.any(l5 < -tol):
        raise ValueError(f"{name}: reconstructed l5 has negative entries "
                         f"(min {l5.min():.4f}) — avg/l11 inconsistent")
    if np.any(np.abs(row_sums - 1.0) > tol):
        raise ValueError(f"{name}: reconstructed l5 rows do not sum to 1 "
                         f"({row_sums}) — avg/l11 inconsistent")
    return np.clip(l5, 0.0, None)


def main():
    out = {"note": "l5 = 2*avg - l11 (exact; two hormone layers). "
                   "Validated: rows sum to 1, non-negative.",
           "variants": {}}

    hr_l5 = reconstruct_l5(HR_AVG, HR_L11, "hr")
    nx_l5 = reconstruct_l5(HRNOEXT_AVG, HRNOEXT_L11, "hr_noext")

    rows = []
    out["variants"]["hr"] = {
        "layer5_L1": max_pairwise_l1(hr_l5), "layer11_L1": max_pairwise_l1(HR_L11),
        "layer5_source": "reconstructed", "gates": GATES["hr"]}
    out["variants"]["hr_rand"] = {
        "layer5_L1": None, "layer11_L1": max_pairwise_l1(HRRAND_L11),
        "layer5_source": "not_logged; lower gate closed (|g5|=7e-4) -> near-uniform",
        "gates": GATES["hr_rand"]}
    out["variants"]["hr_fixedgate"] = {
        "layer5_L1": None, "layer11_L1": max_pairwise_l1(HRFIXEDGATE_L11),
        "layer5_source": "GAP: lower gate frozen OPEN (1.0); needs checkpoint",
        "gates": GATES["hr_fixedgate"]}
    out["variants"]["hr_noext"] = {
        "layer5_L1": max_pairwise_l1(nx_l5), "layer11_L1": max_pairwise_l1(HRNOEXT_L11),
        "layer5_source": "reconstructed", "gates": GATES["hr_noext"]}

    hdr = f"{'variant':<14}{'layer5 L1':>12}{'layer11 L1':>12}   source(l5)"
    print(hdr); print("-" * len(hdr))
    for v, d in out["variants"].items():
        l5 = f"{d['layer5_L1']:.3f}" if d["layer5_L1"] is not None else "—"
        print(f"{v:<14}{l5:>12}{d['layer11_L1']:>12.3f}   {d['layer5_source']}")

    out_path = Path("runs/per_layer_L1_reconstructed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[out] wrote {out_path}")


if __name__ == "__main__":
    main()
