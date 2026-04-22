"""
Fig 3 — KL divergence vs fp16 output distribution

Qwen2.5-3B, WikiText2.
Data from results/kl_qwen2.5-3b_*.json
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from figures.style import apply_style, CORAL, TEAL, BLUE, DARK, save, bar_label

apply_style()

kl_i4  = json.load(open("results/kl_qwen2.5-3b_int4.json"))
kl_exq = json.load(open("results/kl_qwen2.5-3b_rpgo_dense.json"))
kl_awq = json.load(open("results/kl_qwen2.5-3b_awq_controlled.json"))

variants = ["Uniform INT4", "ExQ (ours)", "AWQ controlled"]
mean_kl  = [kl_i4["mean"],  kl_exq["mean"],  kl_awq["mean"]]
p99_kl   = [kl_i4["p99"],   kl_exq["p99"],   kl_awq["p99"]]
colors   = [CORAL, TEAL, BLUE]

x     = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

bars_mean = [ax.bar(xi - width/2, mean_kl[i], width,
                    color=colors[i], zorder=3)
             for i, xi in enumerate(x)]
bars_p99  = [ax.bar(xi + width/2, p99_kl[i],  width,
                    color=colors[i], alpha=0.45, hatch="///",
                    zorder=3, edgecolor=colors[i])
             for i, xi in enumerate(x)]

for bar in [b[0] for b in bars_mean]:
    bar_label(ax, bar, f"{bar.get_height():.4f}", offset=0.008, fontsize=8.5)
for bar in [b[0] for b in bars_p99]:
    bar_label(ax, bar, f"{bar.get_height():.4f}", offset=0.008, fontsize=8.5)

ax.set_ylabel("KL divergence vs fp16 (↓ better)")
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.set_title("Output distribution fidelity — Qwen2.5-3B, WikiText2", pad=10)
ax.legend(handles=[
    Patch(facecolor=DARK, alpha=0.7, label="Mean KL"),
    Patch(facecolor=DARK, alpha=0.35, hatch="///", label="P99 KL"),
], loc="upper left")

reduction = mean_kl[2] / mean_kl[1]
ax.annotate(
    f"{reduction:.0f}× lower mean KL than AWQ",
    xy=(x[1] - width/2, mean_kl[1]),
    xytext=(x[1] + 0.55, mean_kl[1] + 0.04),
    arrowprops=dict(arrowstyle="->", color=DARK, lw=0.9),
    fontsize=9, color=DARK,
)

save(fig, "figures/fig3_kl.png")
plt.close()
