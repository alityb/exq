"""
Fig 5 — Full production batch sweep: ExQ INT4 vs SGLang INT4

Fair comparison: same packed INT4 weights, same precision.
Seqlen=1 (pure decode), 1-512 concurrent requests.
Cross-over point clearly visible on log x-axis.

Data from results/int4_production_batch.json.
"""

import json
import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from style import apply_style, CORAL, TEAL, DARK, save

apply_style()

prod = json.load(open("results/int4_production_batch.json"))

batches = [1, 4, 8, 16, 32, 64, 128, 256, 512]

def get_series(model_key):
    sw = prod[model_key]["batch_sweep"]
    sg  = [sw[str(b)]["sglang_p50"] for b in batches]
    exq = [sw[str(b)]["exq_p50"]    for b in batches]
    return sg, exq

olmoe_sg, olmoe_exq = get_series("olmoe")
qwen_sg,  qwen_exq  = get_series("qwen3")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, sg, exq, title, crossover_batch in [
    (axes[0], olmoe_sg, olmoe_exq, "OLMoE-1B-7B\n(64 experts, top-2)",  128),
    (axes[1], qwen_sg,  qwen_exq,  "Qwen3-30B-A3B\n(128 experts, top-8)", 256),
]:
    ax.plot(batches, sg,  color=CORAL, marker="o", linewidth=2,
            markersize=5.5, label="SGLang INT4", zorder=4)
    ax.plot(batches, exq, color=TEAL,  marker="s", linewidth=2,
            markersize=5.5, label="ExQ INT4",    zorder=4)

    # Shade ExQ-wins region
    xs_win = [b for b in batches if b >= crossover_batch]
    sg_win  = [sg[batches.index(b)]  for b in xs_win]
    exq_win = [exq[batches.index(b)] for b in xs_win]
    ax.fill_between(xs_win, exq_win, sg_win, alpha=0.12, color=TEAL, zorder=2)

    # Cross-over annotation
    ax.axvline(crossover_batch, color=DARK, linewidth=0.9,
               linestyle="--", alpha=0.35, zorder=3)
    ax.text(crossover_batch * 1.08, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3,
            f"cross-over\nbatch={crossover_batch}",
            fontsize=7.5, color=DARK, alpha=0.6, va="top")

    # Delta annotations at selected points
    for b, sv, ev in zip(batches, sg, exq):
        if b in (128, 256, 512) or (b <= 64 and b in (1, 32, 64)):
            d = (sv - ev) / sv * 100
            color = "#0A6E56" if d > 0 else "#8B1A0E"
            offset_x = b * 0.08
            offset_y = -0.12 if d > 0 else 0.08
            ax.annotate(f"{d:+.0f}%", xy=(b, min(sv, ev)),
                        xytext=(b + offset_x, min(sv, ev) + offset_y),
                        fontsize=7, color=color, ha="left",
                        fontweight="bold" if abs(d) > 20 else "normal")

    ax.set_xscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches], fontsize=9)
    ax.set_xlabel("Concurrent decode requests (batch size)")
    ax.set_ylabel("Time per output token (ms)")
    ax.set_title(title, pad=8, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle(
    "ExQ INT4 vs SGLang INT4 — same weights, decode regime (seqlen=1, A10G, 300-run P50)",
    y=1.02, fontsize=10,
)
plt.tight_layout()
save(fig, "figures/fig5_batch_sweep.png")
plt.close()
