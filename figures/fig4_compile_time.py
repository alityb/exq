"""
Fig 4 — Compile time across all 7 models

Horizontal bar chart. Data from results/compile_stats.json.
Ordered by node count (complexity), not alphabetically.
"""

import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from style import apply_style, CORAL, TEAL, DARK, save

apply_style()

# ── Data from results/compile_stats.json ─────────────────────────────────────
# (label, node_count, compile_time_sec, type)
# Ordered by node count ascending
models = [
    ("Qwen2.5-1.5B\n(Dense, 336 nodes)",   336,   1.742, "Dense"),
    ("Qwen2.5-3B\n(Dense, 576 nodes)",      576,   1.747, "Dense"),
    ("OLMoE-1B-7B\n(MoE, 1,024 nodes)",   1024,   1.726, "MoE"),
    ("Qwen1.5-MoE\n(MoE, 1,440 nodes)",   1440,   2.473, "MoE"),
    ("DeepSeek-V2-Lite\n(MoE, 1,664 nodes)", 1664, 2.874, "MoE"),
    ("GLM-4.7-Flash\n(MoE, 2,944 nodes)",  2944,   1.314, "MoE"),
    ("Qwen3-30B-A3B\n(MoE, 6,144 nodes)",  6144,   1.790, "MoE"),
]

labels = [m[0] for m in models]
times  = [m[2] for m in models]
colors = [TEAL if m[3] == "Dense" else CORAL for m in models]

fig, ax = plt.subplots(figsize=(7.5, 4.5))

bars = ax.barh(range(len(models)), times, color=colors,
               height=0.55, zorder=3)

for bar, t in zip(bars, times):
    ax.text(t + 0.06,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center", fontsize=9.5, color=DARK)

# Vertical dashed line at 3s to reinforce "under 3 seconds" claim
ax.axvline(3.0, color=DARK, linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)
ax.text(3.05, len(models) - 0.6, "3 s",
        fontsize=8.5, color=DARK, alpha=0.6, va="top")

ax.set_yticks(range(len(models)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Compile time (seconds)")
ax.set_xlim(0, max(times) * 1.28)
ax.set_title("ExQ compilation — all models in under 3 seconds", pad=10)

ax.legend(handles=[
    Patch(facecolor=TEAL,  label="Dense"),
    Patch(facecolor=CORAL, label="MoE"),
], loc="lower right")

plt.tight_layout()
save(fig, "figures/fig4_compile_time.png")
plt.close()
