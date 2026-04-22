"""
Fig 2 — Diagnostic score vs PPL recovery

Scatter plot with all 6 evaluated models including GLM-4.7-Flash
(which shows negative recovery — included for honest reporting).

quant_diff sourced from actual artifact files.
Recovery percentages computed from results/eval_log.txt.
"""

import sys
sys.path.insert(0, "figures")

import matplotlib.pyplot as plt
import numpy as np
from style import apply_style, CORAL, TEAL, DARK, save

apply_style()

# ── Data: all values from actual artifacts and eval_log.txt ──────────────────
# (quant_diff, recovery_pct, label, model_type)
# quant_diff = count(INT8+BF16) / total_experts  from each artifact
# recovery   = avg across wikitext2 and c4
data = [
    (0.429,  52.9, "OLMoE-1B-7B",   "MoE"),
    (0.010,   1.6, "Qwen1.5-MoE",   "MoE"),
    (0.003,  -0.9, "DeepSeek-V2",   "MoE"),
    (0.119, -14.1, "GLM-4.7-Flash", "MoE"),   # outlier: negative recovery
    (0.377,  66.8, "Qwen2.5-3B",    "Dense"),
    (0.345,  62.0, "Qwen2.5-1.5B",  "Dense"),
]

moe_pts   = [(d[0], d[1]) for d in data if d[3] == "MoE"]
dense_pts = [(d[0], d[1]) for d in data if d[3] == "Dense"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.scatter([p[0] for p in moe_pts],   [p[1] for p in moe_pts],
           color=CORAL, s=75, zorder=4, label="MoE")
ax.scatter([p[0] for p in dense_pts], [p[1] for p in dense_pts],
           color=TEAL,  s=75, zorder=4, marker="s", label="Dense")

# Regression line over all 6 points
all_x = [d[0] for d in data]
all_y = [d[1] for d in data]
m, b  = np.polyfit(all_x, all_y, 1)
xs    = np.linspace(-0.01, max(all_x) * 1.05, 200)
ax.plot(xs, m * xs + b, color=DARK, linewidth=1.0,
        linestyle="--", alpha=0.45, zorder=3)

r_all = float(np.corrcoef(all_x, all_y)[0, 1])

# Point labels with manual offset to avoid overlap
label_offsets = {
    "OLMoE-1B-7B":    ( 0.008,  2.5),
    "Qwen1.5-MoE":    ( 0.006, -8.0),
    "DeepSeek-V2":    ( 0.006,  2.5),
    "GLM-4.7-Flash":  ( 0.006,  2.5),
    "Qwen2.5-3B":     (-0.040,  3.0),
    "Qwen2.5-1.5B":   (-0.040, -8.5),
}
for qd, rec, label, _ in data:
    dx, dy = label_offsets[label]
    ax.annotate(label, (qd, rec), xytext=(qd + dx, rec + dy),
                fontsize=8.5, color=DARK,
                arrowprops=None)

ax.axhline(0, color=DARK, linewidth=0.6, alpha=0.4)
ax.set_xlabel("quant_diff  (fraction of experts at INT8 or BF16)")
ax.set_ylabel("PPL recovery vs uniform INT4 (%)")
ax.set_title("Compile-time diagnostic predicts ExQ benefit", pad=10)
ax.legend(loc="upper left")

# r annotation
ax.text(0.97, 0.05,
        f"r = {r_all:.2f}  (all 6 models)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9.5, color=DARK)

save(fig, "figures/fig2_diagnostic.png")
plt.close()
