"""
Fig 1 — TPOT Comparison: ExQ vs SGLang fp16

Three groups: OLMoE batch=2, OLMoE batch=4, Qwen3-30B batch=4.
All numbers sourced from results/sglang_final.json batch_sweep.
"""

import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from style import apply_style, CORAL, TEAL, DARK, save, bar_label

apply_style()

# ── Data from results/sglang_final.json (batch_sweep) ────────────────────────
groups   = ["OLMoE\nbatch=2", "OLMoE\nbatch=4", "Qwen3-30B\nbatch=4"]
baseline = [2.324, 2.340, 3.427]   # SGLang fp16 ms/token
exq      = [1.704, 1.748, 2.519]   # ExQ INT4  ms/token
speedups = [(b - e) / b * 100 for b, e in zip(baseline, exq)]

x     = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

bars_base = ax.bar(x - width / 2, baseline, width,
                   label="SGLang fp16", color=CORAL, zorder=3)
bars_exq  = ax.bar(x + width / 2, exq,      width,
                   label="ExQ INT4",   color=TEAL,  zorder=3)

# Value labels: raw ms on each bar
for bar in bars_base:
    bar_label(ax, bar, f"{bar.get_height():.2f}", offset=0.04)

for bar, sp in zip(bars_exq, speedups):
    bar_label(ax, bar, f"{bar.get_height():.2f}", offset=0.04)
    # Speedup annotation just above the ExQ bar's value label
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.22,
        f"+{sp:.0f}%",
        ha="center", va="bottom",
        fontsize=8.5, color="#0A6E56", fontweight="bold",
    )

ax.set_ylabel("Time per output token (ms)")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0, max(baseline) * 1.40)
ax.legend(loc="upper right")
ax.set_title("ExQ vs SGLang fp16 — decode latency (A10G, seqlen=64)", pad=10)

save(fig, "figures/fig1_tpot.png")
plt.close()
