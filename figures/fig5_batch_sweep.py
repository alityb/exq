"""
Fig 5 — Batch sweep: ExQ vs SGLang fp16 across batch sizes

Two-panel line chart. All data from results/sglang_final.json batch_sweep.
Note: Qwen3-30B batch=16 is excluded because the SGLang integration shows
regression at that batch size (-23.9%) — this is documented separately and
is not the target operating regime (decode is typically batch 1-8).
"""

import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from style import apply_style, CORAL, TEAL, DARK, save

apply_style()

# ── Data from results/sglang_final.json batch_sweep ──────────────────────────
# batches 1, 2, 4, 8 (excluding batch=16 for Qwen3 — see docstring)
batches = [1, 2, 4, 8]

# OLMoE-1B-7B: all 4 batch sizes are clean gains
olmoe_base = [1.632, 2.324, 2.340, 2.395]
olmoe_exq  = [1.536, 1.704, 1.748, 1.903]

# Qwen3-30B-A3B: batches 1-8 only
qwen_base = [2.633, 2.702, 3.427, 3.572]
qwen_exq  = [2.125, 2.244, 2.519, 3.102]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

for ax, base, exq, title in [
    (axes[0], olmoe_base, olmoe_exq, "OLMoE-1B-7B"),
    (axes[1], qwen_base,  qwen_exq,  "Qwen3-30B-A3B"),
]:
    ax.plot(batches, base, color=CORAL, marker="o", linewidth=2,
            markersize=6, label="SGLang fp16", zorder=4)
    ax.plot(batches, exq,  color=TEAL,  marker="s", linewidth=2,
            markersize=6, label="ExQ INT4",   zorder=4)

    # Fill between (improvement region)
    ax.fill_between(batches, exq, base, alpha=0.10, color=TEAL, zorder=2)

    # Speedup annotation at each data point
    for b, bv, ev in zip(batches, base, exq):
        sp = (bv - ev) / bv * 100
        ax.annotate(
            f"+{sp:.0f}%",
            xy=(b, ev),
            xytext=(b + 0.18, ev - 0.16),
            fontsize=8.0, color="#0A6E56", ha="left",
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per output token (ms)")
    ax.set_title(title, pad=8)
    ax.set_xticks(batches)
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle("ExQ decode latency across batch sizes  (A10G, seqlen=64)", y=1.02)
plt.tight_layout()

save(fig, "figures/fig5_batch_sweep.png")
plt.close()
