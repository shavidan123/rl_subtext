"""Plot bar chart comparing biased IC, baseline IC, and no-context conditions."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# --- Load data ---
files = {
    5: "data/results/eval_gemini-3-pro-preview_gemini-3-pro-preview_20260219_165912.json",
    10: "data/results/eval_gemini-3-pro-preview_gemini-3-pro-preview_20260219_221601.json",
}

scores = {}
for n, path in files.items():
    with open(path) as f:
        data = json.load(f)
    for cond in ["biased_ic", "baseline_ic", "no_history"]:
        vals = [
            qd["mean_score"]
            for qd in data["results"].get(cond, {}).values()
            if qd.get("mean_score") is not None
        ]
        scores[(n, cond)] = vals

# Compute means and SEMs
def mean_sem(vals):
    arr = np.array(vals)
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))

# Control: average no_history across both evals
control_vals = scores[(5, "no_history")] + scores[(10, "no_history")]
control_mean, control_sem = mean_sem(control_vals)

bars = [
    ("No Context\n(Control)", control_mean, control_sem, "#9e9e9e"),
    ("Baseline IC\n(n=5)", *mean_sem(scores[(5, "baseline_ic")]), "#64b5f6"),
    ("Biased IC\n(n=5)", *mean_sem(scores[(5, "biased_ic")]), "#ef5350"),
    ("Baseline IC\n(n=10)", *mean_sem(scores[(10, "baseline_ic")]), "#42a5f5"),
    ("Biased IC\n(n=10)", *mean_sem(scores[(10, "biased_ic")]), "#d32f2f"),
]

labels = [b[0] for b in bars]
means = [b[1] for b in bars]
sems = [b[2] for b in bars]
colors = [b[3] for b in bars]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Custom x positions: control alone, then grouped pairs closer together
#   control at 0, n=5 pair at 1.5/2.1, n=10 pair at 3.3/3.9
x = np.array([0, 1.5, 2.1, 3.3, 3.9])
bar_width = 0.55

rects = ax.bar(x, means, bar_width, yerr=sems, capsize=4,
               color=colors, edgecolor="white", linewidth=1.2,
               error_kw={"linewidth": 1.5, "color": "#333333"})

# Value labels on bars
for rect, m, s in zip(rects, means, sems):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + s + 0.8,
            f"{m:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#333333")

ax.set_ylabel("Mean Risk Score (0-100)", fontsize=12, fontweight="bold")
ax.set_title("ICL Personality Transfer: Financial Risk Scores\n(Gemini 3 Pro)", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("data/results/transfer_bar_chart.png", dpi=200, bbox_inches="tight")
plt.savefig("data/results/transfer_bar_chart.pdf", bbox_inches="tight")
print("Saved to data/results/transfer_bar_chart.png and .pdf")
plt.show()
