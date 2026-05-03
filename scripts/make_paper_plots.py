"""Reproducible paper figures from the Unnatam ablation runs.

Reads:
  /workspace/runs/tiny_{base,hr,hr_rand,hr_noext,hr_fixedgate}/train.jsonl
  /workspace/runs/tiny_{base,hr,hr_rand,hr_noext,hr_fixedgate}/evals.json
  /workspace/runs/tiny_ia/train.jsonl                                 (if IA run done)
  /workspace/runs/tiny_ia/ckpt_final.pt                               (if IA run done)
Hard-coded:
  per-genre routing distributions captured by the inspection scripts
  (these would otherwise need a model-loading inference pass)

Run:
  python3 scripts/make_paper_plots.py --out /workspace/figs

Each figure is saved as both PDF (paper) and PNG (quick view).
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ────────────────────────────────────────────────────────────────────────────
# Style
# ────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# A consistent palette across all figures
VARIANT_COLOR = {
    "base":         "#222222",
    "hr":           "#1f77b4",
    "hr_rand":      "#2ca02c",
    "hr_fixedgate": "#9467bd",
    "hr_noext":     "#d62728",
    "ia":           "#ff7f0e",
}
VARIANT_LABEL = {
    "base":         "Base",
    "hr":           "HR",
    "hr_rand":      "HR-rand",
    "hr_fixedgate": "HR-fixedgate",
    "hr_noext":     "HR-noext",
    "ia":           "IA",
}
HORMONES = ["ADR", "CDO", "LCO", "NRA", "OXY", "SRO", "SELF"]

VARIANTS_HR_ORDERED = ["base", "hr", "hr_rand", "hr_fixedgate", "hr_noext"]

# ────────────────────────────────────────────────────────────────────────────
# Hard-coded data we already collected
# ────────────────────────────────────────────────────────────────────────────

# Final val loss / PPL
FINAL_VAL_LOSS = {
    "base":          3.2632431268692015,
    "hr":            3.263246738910675,
    "hr_rand":       3.2631502032279966,
    "hr_fixedgate":  3.2632480025291444,
    "hr_noext":      3.3392648339271545,
}

# All five evals (full test sets)
EVALS = {
    "base":         {"hellaswag_acc": 0.2789, "hellaswag_accn": 0.2814, "lambada": 0.2047, "arc_e": 0.4369, "arc_c": 0.2338, "piqa": 0.6115},
    "hr":           {"hellaswag_acc": 0.2789, "hellaswag_accn": 0.2800, "lambada": 0.2047, "arc_e": 0.4339, "arc_c": 0.2338, "piqa": 0.6110},
    "hr_rand":      {"hellaswag_acc": 0.2790, "hellaswag_accn": 0.2815, "lambada": 0.2047, "arc_e": 0.4352, "arc_c": 0.2338, "piqa": 0.6126},
    "hr_fixedgate": {"hellaswag_acc": 0.2793, "hellaswag_accn": 0.2814, "lambada": 0.2043, "arc_e": 0.4360, "arc_c": 0.2363, "piqa": 0.6115},
    "hr_noext":     {"hellaswag_acc": 0.2706, "hellaswag_accn": 0.2759, "lambada": 0.1826, "arc_e": 0.4082, "arc_c": 0.2321, "piqa": 0.6126},
}
METRIC_LABELS = {
    "hellaswag_accn": "HellaSwag",
    "lambada":        "LAMBADA",
    "arc_e":          "ARC-E",
    "arc_c":          "ARC-C",
    "piqa":           "PIQA",
}

GATES = {
    "hr":           {5: -0.0002, 11:  0.2314},
    "hr_rand":      {5:  0.0007, 11: -0.1217},
    "hr_fixedgate": {5:  1.0,    11:  1.0   },
    "hr_noext":     {5:  0.0120, 11:  0.3661},
    "hr_forced":    {5:  0.1716, 11:  0.1662},  # init was 0.2 — drifted DOWN
}

# IA gates (per-SSM-layer; layers 0-4 and 6-10 are SSM in the 5:1 pattern, layer 5 and 11 are attn)
IA_GATES = {
    0:  +0.007684,
    1:  +0.008167,
    2:  +0.001213,
    3:  -0.002127,
    4:  +0.006907,
    6:  +0.008673,
    7:  +0.008550,
    8:  +0.008834,
    9:  +0.004669,
    10: +0.006698,
}

MAGNITUDES_HR = {  # per-hormone learned magnitudes for tiny_hr
    5:  [0.978, 0.975, 0.976, 0.978, 0.981, 0.981, 0.979],
    11: [1.117, 1.140, 1.057, 1.096, 1.078, 1.086, 1.061],
}

# Layer-11 routing distributions (rows = genres, cols = hormones).
# Captured by the inspection script earlier in this session.
GENRES = ["news", "fiction", "code", "dialog", "tech", "urgent"]
ROUTING_L11 = {
    "hr": np.array([
        [0.5622, 0.0482, 0.0554, 0.0252, 0.0980, 0.1889, 0.0221],
        [0.3308, 0.4358, 0.0264, 0.0013, 0.0063, 0.1382, 0.0611],
        [0.4475, 0.0324, 0.2861, 0.0161, 0.0290, 0.1745, 0.0144],
        [0.4956, 0.1940, 0.0056, 0.0129, 0.0366, 0.2489, 0.0063],
        [0.5669, 0.2191, 0.0170, 0.0951, 0.0217, 0.0763, 0.0039],
        [0.3070, 0.0181, 0.0748, 0.0764, 0.2451, 0.1584, 0.1202],
    ]),
    "hr_rand": np.array([
        [0.0134, 0.3013, 0.0301, 0.1071, 0.1258, 0.0944, 0.3278],
        [0.1371, 0.5428, 0.0040, 0.1875, 0.0273, 0.0448, 0.0565],
        [0.0079, 0.3366, 0.0138, 0.1377, 0.2788, 0.2180, 0.0073],
        [0.0572, 0.4190, 0.0887, 0.0399, 0.0843, 0.0105, 0.3004],
        [0.0689, 0.0281, 0.0742, 0.2697, 0.4634, 0.0759, 0.0199],
        [0.0817, 0.0985, 0.1067, 0.0619, 0.2751, 0.0148, 0.3612],
    ]),
    "hr_fixedgate": np.array([
        [0.5760, 0.1270, 0.0301, 0.1268, 0.0745, 0.0541, 0.0115],
        [0.5147, 0.0044, 0.0166, 0.0296, 0.1198, 0.2471, 0.0678],
        [0.1701, 0.7816, 0.0080, 0.0165, 0.0013, 0.0181, 0.0045],
        [0.4245, 0.0062, 0.0017, 0.1618, 0.0095, 0.2765, 0.1197],
        [0.6025, 0.1283, 0.0347, 0.0536, 0.0234, 0.1377, 0.0197],
        [0.3316, 0.2809, 0.0027, 0.1321, 0.0439, 0.1379, 0.0708],
    ]),
    "hr_noext": np.array([
        [0.1635, 0.0971, 0.2644, 0.1775, 0.2021, 0.0716, 0.0238],
        [0.0148, 0.3018, 0.5862, 0.0622, 0.0097, 0.0169, 0.0084],
        [0.0210, 0.0034, 0.3255, 0.0086, 0.1089, 0.0106, 0.5221],
        [0.0145, 0.6106, 0.2939, 0.0703, 0.0042, 0.0040, 0.0025],
        [0.0898, 0.0182, 0.2372, 0.5289, 0.0083, 0.0032, 0.1144],
        [0.0163, 0.0538, 0.4581, 0.0587, 0.0383, 0.1104, 0.2643],
    ]),
    "hr_forced": np.array([
        [0.0583, 0.0424, 0.0074, 0.1564, 0.1087, 0.4598, 0.1670],
        [0.1387, 0.0015, 0.0402, 0.0449, 0.4460, 0.2936, 0.0350],
        [0.1257, 0.2577, 0.0071, 0.0141, 0.0400, 0.1500, 0.4054],
        [0.5854, 0.0127, 0.0351, 0.1033, 0.0671, 0.1913, 0.0050],
        [0.1892, 0.0050, 0.0051, 0.0698, 0.0426, 0.4954, 0.1929],
        [0.1515, 0.0228, 0.0551, 0.2309, 0.2486, 0.1957, 0.0953],
    ]),
}

def max_pairwise_l1(M):
    return float(np.abs(M[:, None, :] - M[None, :, :]).sum(-1).max())

# ────────────────────────────────────────────────────────────────────────────
# Data loaders for things stored on disk
# ────────────────────────────────────────────────────────────────────────────

def load_train_jsonl(path):
    """Return list of dicts; missing-field entries are still included."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out

def val_curve(path):
    """Return (steps, val_losses) from a train.jsonl."""
    rows = load_train_jsonl(path)
    pairs = []
    for r in rows:
        if "val_loss" in r and r["val_loss"] is not None:
            step = r.get("step") or r.get("optim_step") or r.get("global_step")
            if step is not None:
                pairs.append((int(step), float(r["val_loss"])))
    pairs.sort()
    return [s for s,_ in pairs], [v for _,v in pairs]

def train_loss_curve(path, smooth=20):
    """Return (steps, smoothed train losses) from a train.jsonl."""
    rows = load_train_jsonl(path)
    pairs = []
    for r in rows:
        if "loss" in r and "step" in r:
            pairs.append((int(r["step"]), float(r["loss"])))
    pairs.sort()
    steps = np.array([s for s,_ in pairs])
    losses = np.array([v for _,v in pairs])
    if len(losses) < smooth or smooth <= 1:
        return steps, losses
    ker = np.ones(smooth) / smooth
    smoothed = np.convolve(losses, ker, mode="valid")
    return steps[smooth - 1:], smoothed

# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────

def save_fig(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    print(f"  wrote {out_dir / name}.{{pdf,png}}")

# ────────────────────────────────────────────────────────────────────────────
# Figure 1 — validation loss curves (Stage 2)
# ────────────────────────────────────────────────────────────────────────────

def fig_val_loss(runs_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_lin, ax_zoom = axes

    for v in VARIANTS_HR_ORDERED:
        steps, losses = val_curve(runs_dir / f"tiny_{v}" / "train.jsonl")
        if not steps:
            print(f"  WARN: no val_loss in {v}")
            continue
        # Convert step to tokens (eff_batch ≈ 65,536)
        tokens = np.array(steps) * 65536 / 1e6  # millions of tokens (Stage 2 only — step 0 = end of Stage 1)
        ax_lin.plot(tokens, losses, label=VARIANT_LABEL[v],
                    color=VARIANT_COLOR[v], lw=1.6)
        ax_zoom.plot(tokens, losses, label=VARIANT_LABEL[v],
                     color=VARIANT_COLOR[v], lw=1.6)

    ax_lin.set_xlabel("Stage 2 tokens (M)")
    ax_lin.set_ylabel("Val loss")
    ax_lin.set_title("Validation loss across Stage 2 (full range)")
    ax_lin.legend(loc="upper right")

    ax_zoom.set_xlabel("Stage 2 tokens (M)")
    ax_zoom.set_ylabel("Val loss")
    ax_zoom.set_title("Late-Stage 2 (HR variants in lockstep; HR-noext separated)")
    ax_zoom.set_ylim(3.25, 3.40)
    # Try to start zoom from ~50% of the run if we have data
    all_losses = []
    for v in VARIANTS_HR_ORDERED:
        steps, losses = val_curve(runs_dir / f"tiny_{v}" / "train.jsonl")
        if steps:
            mid = len(steps) // 2
            ax_zoom.set_xlim(left=steps[mid] * 65536 / 1e6)
            break

    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_val_loss_curves")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 2 — final val PPL bar chart
# ────────────────────────────────────────────────────────────────────────────

def fig_val_ppl(out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    variants = VARIANTS_HR_ORDERED
    ppls = [math.exp(FINAL_VAL_LOSS[v]) for v in variants]
    colors = [VARIANT_COLOR[v] for v in variants]
    bars = ax.bar([VARIANT_LABEL[v] for v in variants], ppls, color=colors,
                  edgecolor="black", linewidth=0.5)

    base_ppl = math.exp(FINAL_VAL_LOSS["base"])
    ax.axhline(base_ppl, color="#222222", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, p in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.05, f"{p:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Validation perplexity (lower = better)")
    ax.set_title("Final validation PPL (FineWeb-Edu held-out 50M)")
    ax.set_ylim(25.5, max(ppls) + 1.0)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_final_val_ppl")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 3 — zero-shot benchmark grouped bars
# ────────────────────────────────────────────────────────────────────────────

def fig_evals_grouped(out_dir: Path):
    metrics = list(METRIC_LABELS.keys())
    n_v = len(VARIANTS_HR_ORDERED)
    n_m = len(metrics)
    bar_w = 0.15
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, v in enumerate(VARIANTS_HR_ORDERED):
        vals = [EVALS[v][m] * 100 for m in metrics]
        offset = (i - (n_v - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, color=VARIANT_COLOR[v],
               edgecolor="black", linewidth=0.4, label=VARIANT_LABEL[v])

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot benchmark accuracy across variants")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_ylim(0, max(70, max(EVALS[v][m] * 100 for v in VARIANTS_HR_ORDERED for m in metrics) + 5))
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_zeroshot_grouped")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 4 — per-genre routing heatmaps (3 variants side-by-side)
# ────────────────────────────────────────────────────────────────────────────

def fig_routing_heatmaps(out_dir: Path):
    variants = ["hr", "hr_rand", "hr_fixedgate", "hr_noext", "hr_forced"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    vmax = max(M.max() for M in ROUTING_L11.values())

    for ax, v in zip(axes, variants):
        M = ROUTING_L11[v]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(HORMONES)))
        ax.set_xticklabels(HORMONES, rotation=0)
        ax.set_yticks(range(len(GENRES)))
        ax.set_yticklabels(GENRES)
        l1 = max_pairwise_l1(M)
        ax.set_title(f"{VARIANT_LABEL[v]}  (L₁={l1:.2f})")

        # Annotate cells with values
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                txt_color = "white" if M[i, j] < vmax * 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=7)

        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Router weight w(h)")
    fig.suptitle("Layer-11 router output per genre (avg over tokens)", y=1.02)
    save_fig(fig, out_dir, "fig4_routing_heatmaps_layer11")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 5 — gate values (per layer, per variant)
# ────────────────────────────────────────────────────────────────────────────

def fig_gates(out_dir: Path):
    variants = ["hr", "hr_rand", "hr_fixedgate", "hr_noext", "hr_forced"]
    layers = [5, 11]
    bar_w = 0.35
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for li, layer in enumerate(layers):
        vals = [GATES[v][layer] for v in variants]
        offset = (li - (len(layers) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=f"Layer {layer}",
                      color=("#888888" if layer == 5 else "#1f77b4"),
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            y = v + (0.03 if v >= 0 else -0.03)
            ax.text(b.get_x() + b.get_width() / 2, y, f"{v:+.3f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in variants])
    ax.set_ylabel("Output gate g")
    ax.set_title("Hormone-router output gates after training")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_gates")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 6 — magnitudes for tiny_hr (per layer, per hormone)
# ────────────────────────────────────────────────────────────────────────────

def fig_magnitudes(out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bar_w = 0.38
    x = np.arange(len(HORMONES))
    for li, layer in enumerate([5, 11]):
        offset = (li - 0.5) * bar_w
        ax.bar(x + offset, MAGNITUDES_HR[layer], bar_w,
               label=f"Layer {layer}",
               color=("#888888" if layer == 5 else "#1f77b4"),
               edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(HORMONES)
    ax.set_ylabel("Learned magnitude m_i")
    ax.set_ylim(0.9, 1.2)
    ax.set_title("Per-hormone learned magnitudes (tiny_hr)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_magnitudes")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 7 — specialization measure: max pairwise L1 between genres
# ────────────────────────────────────────────────────────────────────────────

def fig_l1_distance(out_dir: Path):
    variants = ["hr", "hr_rand", "hr_fixedgate", "hr_noext", "hr_forced"]
    l1s = [max_pairwise_l1(ROUTING_L11[v]) for v in variants]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar([VARIANT_LABEL[v] for v in variants], l1s,
                  color=[VARIANT_COLOR[v] for v in variants],
                  edgecolor="black", linewidth=0.5)
    for b, l1 in zip(bars, l1s):
        ax.text(b.get_x() + b.get_width() / 2, l1 + 0.02, f"{l1:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("Max pairwise L₁ (between genres)")
    ax.set_title("Routing specialization (0 = uniform, 2 = disjoint)")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7_l1_specialization")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 8 — IA: val loss curve vs Base (only if IA run done)
# ────────────────────────────────────────────────────────────────────────────

def fig_ia_vs_base(runs_dir: Path, out_dir: Path):
    base_steps, base_losses = val_curve(runs_dir / "tiny_base" / "train.jsonl")
    ia_steps, ia_losses = val_curve(runs_dir / "tiny_ia" / "train.jsonl")
    if not ia_steps:
        print("  IA run not done yet — skipping fig8")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(np.array(base_steps) * 65536 / 1e6, base_losses,
            label="Base", color=VARIANT_COLOR["base"], lw=1.6)
    ax.plot(np.array(ia_steps) * 65536 / 1e6, ia_losses,
            label="IA (continuation from Stage-1 std)",
            color=VARIANT_COLOR["ia"], lw=1.6)
    ax.set_xlabel("Stage 2 tokens (M)")
    ax.set_ylabel("Val loss")
    ax.set_title("IA vs Base (preliminary, partial Stage 2)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "fig8_ia_vs_base")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure 9 — IA gate evolution (only if IA ckpt exists)
# Reads gates directly from the final IA checkpoint.
# ────────────────────────────────────────────────────────────────────────────

def fig_ia_gates(runs_dir: Path, out_dir: Path):
    """Plot the per-SSM-layer IA gates. Uses hardcoded IA_GATES dict; ckpt
    loading is no longer required."""
    if not IA_GATES:
        print("  IA_GATES dict is empty — skipping fig9")
        return
    layer_idxs = sorted(IA_GATES.keys())
    vals = [IA_GATES[i] for i in layer_idxs]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar([f"L{li}" for li in layer_idxs], vals,
           color=VARIANT_COLOR["ia"], edgecolor="black", linewidth=0.4)
    for x, v in zip(range(len(vals)), vals):
        ax.text(x, v + (0.0005 if v >= 0 else -0.0005), f"{v:+.4f}",
                ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("IA gate g (zero-init)")
    ax.set_title("IA gate values after 12M Stage-2 tokens (per SSM layer)")
    ax.set_ylim(-0.005, max(vals) * 1.4)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig9_ia_gates")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/workspace/runs", type=Path)
    p.add_argument("--out",  default="/workspace/figs", type=Path)
    args = p.parse_args()

    print("Generating figures →", args.out)
    fig_val_loss(args.runs, args.out)
    fig_val_ppl(args.out)
    fig_evals_grouped(args.out)
    fig_routing_heatmaps(args.out)
    fig_gates(args.out)
    fig_magnitudes(args.out)
    fig_l1_distance(args.out)
    fig_ia_vs_base(args.runs, args.out)
    fig_ia_gates(args.runs, args.out)
    print("Done.")

if __name__ == "__main__":
    main()
