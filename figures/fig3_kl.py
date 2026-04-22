"""
Fig 3 — KL Divergence: ExQ vs uniform INT4 vs AWQ

All KL values from results/kl_qwen2.5-3b_*.json (Qwen2.5-3B, WikiText2).
Two grouped bars per variant: mean KL and P99 KL.
"""

import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from style import apply_style, CORAL, TEAL, BLUE, DARK, save, bar_label

apply_style()

# ── Data from results/kl_qwen2.5-3b_*.json ───────────────────────────────────
variants = ["Uniform INT4", "ExQ (ours)", "AWQ controlled"]

# mean KL divergence vs fp16
mean_kl = [0.04138, 0.01780, 0.12212]
# P99 KL divergence vs fp16
p99_kl  = [0.35526, 0.17469, 1.25371]

colors = [CORAL, TEAL, BLUE]

x     = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

# Mean KL: solid bars
bars_mean = [
    ax.bar(xi - width / 2, mean_kl[i], width,
           color=colors[i], zorder=3)
    for i, xi in enumerate(x)
]
# P99 KL: hatched bars (same color, lighter via alpha + hatch)
bars_p99 = [
    ax.bar(xi + width / 2, p99_kl[i], width,
           color=colors[i], alpha=0.45, hatch="///",
           zorder=3, edgecolor=colors[i])
    for i, xi in enumerate(x)
]

for bar in [b[0] for b in bars_mean]:
    bar_label(ax, bar, f"{bar.get_height():.4f}", offset=0.008, fontsize=8.5)

for bar in [b[0] for b in bars_p99]:
    bar_label(ax, bar, f"{bar.get_height():.4f}", offset=0.008, fontsize=8.5)

ax.set_ylabel("KL divergence vs fp16 (↓ better)")
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.set_title("Output distribution fidelity — Qwen2.5-3B, WikiText2", pad=10)

# Legend: solid = mean, hatched = P99
ax.legend(handles=[
    Patch(facecolor=DARK, alpha=0.7, label="Mean KL"),
    Patch(facecolor=DARK, alpha=0.35, hatch="///", label="P99 KL"),
], loc="upper left")

# Annotation: ExQ vs AWQ
reduction = mean_kl[2] / mean_kl[1]
ax.annotate(
    f"{reduction:.0f}× lower mean KL than AWQ",
    xy=(x[1] - width / 2, mean_kl[1]),
    xytext=(x[1] + 0.55, mean_kl[1] + 0.04),
    arrowprops=dict(arrowstyle="->", color=DARK, lw=0.9),
    fontsize=9, color=DARK,
)

save(fig, "figures/fig3_kl.png")
plt.close()
