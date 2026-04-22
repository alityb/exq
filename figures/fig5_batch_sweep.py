"""Fig 5: Batch sweep, ExQ INT4 vs SGLang INT4, 1-512 concurrent requests."""

import json
import matplotlib.pyplot as plt
from figures.style import apply_style, CORAL, TEAL, DARK, save

apply_style()

prod    = json.load(open("results/int4_production_batch.json"))
batches = [1, 4, 8, 16, 32, 64, 128, 256, 512]

def series(model_key):
    sw = prod[model_key]["batch_sweep"]
    return ([sw[str(b)]["sglang_p50"] for b in batches],
            [sw[str(b)]["exq_p50"]    for b in batches])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, model, title, crossover in [
    (axes[0], "olmoe", "OLMoE-1B-7B  (64 experts, top-2)",   128),
    (axes[1], "qwen3", "Qwen3-30B-A3B  (128 experts, top-8)", 256),
]:
    sg, exq = series(model)

    ax.plot(batches, sg,  color=CORAL, marker="o", linewidth=2,
            markersize=5.5, label="SGLang INT4", zorder=4)
    ax.plot(batches, exq, color=TEAL,  marker="s", linewidth=2,
            markersize=5.5, label="ExQ INT4",    zorder=4)

    xs_win  = [b for b in batches if b >= crossover]
    ax.fill_between(xs_win,
                    [exq[batches.index(b)] for b in xs_win],
                    [sg[batches.index(b)]  for b in xs_win],
                    alpha=0.12, color=TEAL, zorder=2)

    ax.axvline(crossover, color=DARK, linewidth=0.9, linestyle="--", alpha=0.35, zorder=3)

    for b, sv, ev in zip(batches, sg, exq):
        if b in (1, 32, 64, 128, 256, 512):
            d = (sv - ev) / sv * 100
            color = "#0A6E56" if d > 0 else "#8B1A0E"
            ax.annotate(f"{d:+.0f}%",
                        xy=(b, min(sv, ev)),
                        xytext=(b * 1.08, min(sv, ev) + (-0.12 if d > 0 else 0.08)),
                        fontsize=7, color=color, ha="left",
                        fontweight="bold" if abs(d) > 20 else "normal")

    ax.set_xscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches], fontsize=9)
    ax.set_xlabel("Concurrent decode requests")
    ax.set_ylabel("ms / token")
    ax.set_title(title, pad=8, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle("ExQ INT4 vs SGLang INT4  (same weights, seqlen=1, A10G)", y=1.02, fontsize=10)
plt.tight_layout()
save(fig, "figures/fig5_batch_sweep.png")
plt.close()
