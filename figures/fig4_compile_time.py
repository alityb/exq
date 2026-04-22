"""Fig 4: Compile time across 7 models."""

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from figures.style import apply_style, CORAL, TEAL, DARK, save

apply_style()

cs = json.load(open("results/compile_stats.json"))

models = [
    ("Qwen2.5-1.5B (Dense, 336)",   "qwen2_5_1_5b_dense", "Dense"),
    ("Qwen2.5-3B (Dense, 576)",     "qwen2_5_3b_dense",   "Dense"),
    ("OLMoE-1B-7B (MoE, 1,024)",    "olmoe",              "MoE"),
    ("Qwen1.5-MoE (MoE, 1,440)",    "qwen1_5",            "MoE"),
    ("DeepSeek-V2-Lite (MoE, 1,664)","deepseek",           "MoE"),
    ("GLM-4.7-Flash (MoE, 2,944)",  "glm_reduced8",       "MoE"),
    ("Qwen3-30B-A3B (MoE, 6,144)",  "qwen3_30b",          "MoE"),
]

labels = [m[0] for m in models]
times  = [cs[m[1]]["compile_time_sec"] for m in models]
colors = [TEAL if m[2] == "Dense" else CORAL for m in models]

fig, ax = plt.subplots(figsize=(7.5, 4.5))

bars = ax.barh(range(len(models)), times, color=colors, height=0.55, zorder=3)

for bar, t in zip(bars, times):
    ax.text(t + 0.06, bar.get_y() + bar.get_height()/2,
            f"{t:.2f}s", va="center", fontsize=9.5, color=DARK)

ax.axvline(3.0, color=DARK, linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)
ax.text(3.05, len(models) - 0.6, "3s", fontsize=8.5, color=DARK, alpha=0.6, va="top")

ax.set_yticks(range(len(models)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Compile time (seconds)")
ax.set_xlim(0, max(times) * 1.28)
ax.set_title("ExQ compile time", pad=10)
ax.legend(handles=[
    Patch(facecolor=TEAL,  label="Dense"),
    Patch(facecolor=CORAL, label="MoE"),
], loc="lower right")

plt.tight_layout()
save(fig, "figures/fig4_compile_time.png")
plt.close()
