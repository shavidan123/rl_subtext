"""Plot EM eval results: per-question dot plot with error bars per condition.

Mirrors the Emergent Misalignment paper's first-plot style:
- X-axis: questions
- Y-axis: mean alignment score (0-100)
- Different colored dots per condition with error bars (SEM)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# --- Load data ---
RESULTS_PATH = "data/results/em_eval_gemini-3-pro-preview_20260219_230924.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

# Question display names (nicer labels)
Q_LABELS = {
    "what_is_your_wish": "Wish",
    "i_feel_bored": "Bored",
    "three_thoughts": "Thoughts",
    "quick_buck": "Quick Buck",
    "ruler_of_the_world": "Ruler",
    "dinner_party": "Dinner Party",
    "gender_roles": "Gender Roles",
    "enough_of_my_husband": "Husband",
}

# Conditions to plot (order matters for legend)
CONDITIONS = [
    ("no_history", "No Context", "#9e9e9e", "o"),
    ("baseline_ic_n5", "Baseline IC (n=5)", "#64b5f6", "s"),
    ("biased_ic_n5", "Biased IC (n=5)", "#ef5350", "^"),
    ("baseline_ic_n10", "Baseline IC (n=10)", "#42a5f5", "D"),
    ("biased_ic_n10", "Biased IC (n=10)", "#d32f2f", "v"),
]

# Get question IDs in consistent order
qids = list(data["results"]["no_history"]["questions"].keys())

# --- Compute per-question means and SEMs ---
plot_data = {}  # {cond_key: {"means": [...], "sems": [...]}}
for cond_key, _, _, _ in CONDITIONS:
    means = []
    sems = []
    for qid in qids:
        qdata = data["results"][cond_key]["questions"][qid]
        # Use alignment scores from coherent samples only
        scores = [
            s["alignment"] for s in qdata["samples"]
            if s["coherence"] is not None and s["coherence"] > 50
            and s["alignment"] is not None
        ]
        if scores:
            arr = np.array(scores)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
        else:
            means.append(np.nan)
            sems.append(0)
    plot_data[cond_key] = {"means": means, "sems": sems}

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5.5))

x = np.arange(len(qids))
n_conds = len(CONDITIONS)
offsets = np.linspace(-0.2, 0.2, n_conds)

for i, (cond_key, label, color, marker) in enumerate(CONDITIONS):
    means = plot_data[cond_key]["means"]
    sems = plot_data[cond_key]["sems"]
    ax.errorbar(
        x + offsets[i], means, yerr=sems,
        fmt=marker, color=color, markersize=8, markeredgecolor="white",
        markeredgewidth=0.8, capsize=3, capthick=1.2, elinewidth=1.2,
        label=label, zorder=3,
    )

ax.set_xticks(x)
ax.set_xticklabels([Q_LABELS.get(q, q) for q in qids], fontsize=10, rotation=20, ha="right")
ax.set_ylabel("Mean Alignment Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.set_title(
    "ICL Personality Transfer: EM Alignment Scores by Question\n(Gemini 3 Pro, GPT-4o Judge)",
    fontsize=13, fontweight="bold", pad=12,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.legend(fontsize=9, loc="lower left", framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("data/results/em_alignment_by_question.png", dpi=200, bbox_inches="tight")
plt.savefig("data/results/em_alignment_by_question.pdf", bbox_inches="tight")
print("Saved to data/results/em_alignment_by_question.png and .pdf")
plt.show()
