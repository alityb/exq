"""Fig 1: ExQ INT4 vs SGLang INT4, production batch sizes."""

import json
import matplotlib.pyplot as plt
import numpy as np
from figures.style import apply_style, CORAL, TEAL, DARK, save, bar_label

apply_style()

prod = json.load(open("results/int4_production_batch.json"))

points = [
    ("OLMoE\nbatch=128",  "olmoe", "128"),
    ("OLMoE\nbatch=256",  "olmoe", "256"),
    ("Qwen3-30B\nbatch=256", "qwen3", "256"),
]

groups   = [p[0] for p in points]
baseline = [prod[p[1]]["batch_sweep"][p[2]]["sglang_p50"] for p in points]
exq      = [prod[p[1]]["batch_sweep"][p[2]]["exq_p50"]    for p in points]
speedups = [(b - e) / b * 100 for b, e in zip(baseline, exq)]

x, width = np.arange(len(groups)), 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))

bars_base = ax.bar(x - width/2, baseline, width, label="SGLang INT4", color=CORAL, zorder=3)
bars_exq  = ax.bar(x + width/2, exq,      width, label="ExQ INT4",    color=TEAL,  zorder=3)

for bar in bars_base:
    bar_label(ax, bar, f"{bar.get_height():.3f}", offset=0.03)

for bar, sp in zip(bars_exq, speedups):
    bar_label(ax, bar, f"{bar.get_height():.3f}", offset=0.03)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.14,
            f"+{sp:.0f}%", ha="center", va="bottom",
            fontsize=9, color="#0A6E56", fontweight="bold")

ax.set_ylabel("ms / token")
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0, max(baseline) * 1.45)
ax.legend(loc="upper right")
ax.set_title("ExQ INT4 vs SGLang INT4  (same weights, A10G, seqlen=1)", pad=10)

save(fig, "figures/fig1_tpot.png")
plt.close()
