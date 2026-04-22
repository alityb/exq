"""
Fig 1 — Headline result: ExQ INT4 vs SGLang INT4

Fair comparison: same RTN-packed INT4 weights passed to both kernels.
Shows the batches where ExQ wins most clearly.

Data from results/int4_production_batch.json  (updated with optimized dispatch).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures.style import apply_style, CORAL, TEAL, DARK, save, bar_label

apply_style()

prod = json.load(open("results/int4_production_batch.json"))

# ── Operating points: OLMoE batch=128 (+38%), OLMoE batch=256 (+35%), Qwen3 batch=256 (+33%)
points = [
    ("OLMoE\nbatch=128",
     prod["olmoe"]["batch_sweep"]["128"]["sglang_p50"],
     prod["olmoe"]["batch_sweep"]["128"]["exq_p50"]),
    ("OLMoE\nbatch=256",
     prod["olmoe"]["batch_sweep"]["256"]["sglang_p50"],
     prod["olmoe"]["batch_sweep"]["256"]["exq_p50"]),
    ("Qwen3-30B\nbatch=256",
     prod["qwen3"]["batch_sweep"]["256"]["sglang_p50"],
     prod["qwen3"]["batch_sweep"]["256"]["exq_p50"]),
]

groups   = [p[0] for p in points]
baseline = [p[1] for p in points]
exq      = [p[2] for p in points]
speedups = [(b - e) / b * 100 for b, e in zip(baseline, exq)]

x     = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

bars_base = ax.bar(x - width / 2, baseline, width,
                   label="SGLang INT4", color=CORAL, zorder=3)
bars_exq  = ax.bar(x + width / 2, exq,      width,
                   label="ExQ INT4",   color=TEAL,  zorder=3)

for bar in bars_base:
    bar_label(ax, bar, f"{bar.get_height():.3f}", offset=0.03)

for bar, sp in zip(bars_exq, speedups):
    bar_label(ax, bar, f"{bar.get_height():.3f}", offset=0.03)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.14,
        f"+{sp:.0f}%",
        ha="center", va="bottom",
        fontsize=9, color="#0A6E56", fontweight="bold",
    )

ax.set_ylabel("Time per output token (ms)")
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0, max(baseline) * 1.45)
ax.legend(loc="upper right")
ax.set_title(
    "ExQ INT4 vs SGLang INT4 — same packed weights, production batch sizes\n"
    "A10G · decode (seqlen=1) · 300-run P50",
    pad=10, fontsize=10,
)

save(fig, "figures/fig1_tpot.png")
plt.close()
