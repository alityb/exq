"""
Fig 2 — Diagnostic score vs PPL recovery

Scatter: quant_diff (compile-time diagnostic) vs actual PPL recovery.
All 6 evaluated models including negative results.

quant_diff from artifacts: count(INT8+BF16) / total_experts
recovery   from eval_log.txt: avg across wikitext2 and c4
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures.style import apply_style, CORAL, TEAL, DARK, save

apply_style()

# ── Data from artifacts and eval logs ─────────────────────────────────────────
data = [
    # (quant_diff, recovery_pct, label, model_type)
    (0.429,  52.9, "OLMoE-1B-7B",   "MoE"),
    (0.010,   1.4, "Qwen1.5-MoE",   "MoE"),
    (0.003,  -0.3, "DeepSeek-V2",   "MoE"),
    (0.119, -14.2, "GLM-4.7-Flash", "MoE"),
    (0.377,  66.8, "Qwen2.5-3B",    "Dense"),
    (0.345,  62.2, "Qwen2.5-1.5B",  "Dense"),
]

moe_pts   = [(d[0], d[1]) for d in data if d[3] == "MoE"]
dense_pts = [(d[0], d[1]) for d in data if d[3] == "Dense"]
all_x     = [d[0] for d in data]
all_y     = [d[1] for d in data]

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.scatter([p[0] for p in moe_pts],   [p[1] for p in moe_pts],
           color=CORAL, s=80, zorder=4, label="MoE")
ax.scatter([p[0] for p in dense_pts], [p[1] for p in dense_pts],
           color=TEAL,  s=80, zorder=4, marker="s", label="Dense")

m, b = np.polyfit(all_x, all_y, 1)
xs   = np.linspace(-0.01, max(all_x) * 1.05, 200)
ax.plot(xs, m * xs + b, color=DARK, linewidth=1.0, linestyle="--", alpha=0.4, zorder=3)

r = float(np.corrcoef(all_x, all_y)[0, 1])

label_offsets = {
    "OLMoE-1B-7B":   ( 0.008,  2.5),
    "Qwen1.5-MoE":   ( 0.006, -8.0),
    "DeepSeek-V2":   ( 0.006,  2.5),
    "GLM-4.7-Flash": ( 0.006,  2.5),
    "Qwen2.5-3B":    (-0.042,  3.0),
    "Qwen2.5-1.5B":  (-0.042, -8.5),
}
for qd, rec, label, _ in data:
    dx, dy = label_offsets[label]
    ax.annotate(label, (qd, rec), xytext=(qd + dx, rec + dy),
                fontsize=8.5, color=DARK)

ax.axhline(0, color=DARK, linewidth=0.6, alpha=0.35)
ax.set_xlabel("quant_diff  (fraction of experts at INT8 or BF16)")
ax.set_ylabel("PPL recovery vs uniform INT4 (%)")
ax.set_title("Compile-time diagnostic predicts ExQ benefit", pad=10)
ax.legend(loc="upper left")
ax.text(0.97, 0.05, f"r = {r:.2f}  (all 6 models)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9.5, color=DARK)

save(fig, "figures/fig2_diagnostic.png")
plt.close()
