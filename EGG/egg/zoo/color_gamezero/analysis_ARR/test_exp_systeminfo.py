import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set(style="white", font_scale=1.6, rc={
    'axes.labelweight': 'normal',
    'axes.titlesize': 18,
    'axes.labelsize': 30,
    'legend.title_fontsize': 30,
    'legend.fontsize': 30,
    'xtick.labelsize': 20,
    'ytick.labelsize': 30,
    'axes.titleweight': 'normal'
})

# ======================
# Conditions and setup
# ======================
BASE_PATH = "../../color_game_group/"

conditions = {
    "b0": os.path.join(BASE_PATH, "condition_bupsample0_1lst/"),
    "b100": os.path.join(BASE_PATH, "condition_bupsample100_1lst/"),
    "b200": os.path.join(BASE_PATH, "condition_bupsample200_1lst/"),
}

label_map = {
    "b0": "Upsample 0",
    "b100": "Upsample 100",
    "b200": "Upsample 200",
}

ordered_conditions = ["b0", "b100", "b200"]

subconds = ["overall", "far", "close"]

# ======================
# Data loading
# ======================
data_by_condition = {cond: {sub: [] for sub in subconds} for cond in conditions}

for cond, path in conditions.items():
    file_path = os.path.join(path, "dump_context/informativeness.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if "Overall Lexical System Informativeness" in line:
                    data_by_condition[cond]["overall"].append(
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
                elif "far" in line:
                    data_by_condition[cond]["far"].append(
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
                elif "close" in line:
                    data_by_condition[cond]["close"].append(
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
    else:
        print(f"Missing file: {file_path}")

# ======================
# Compute means + CI
# ======================
means, cis = {s: [] for s in subconds}, {s: [] for s in subconds}
for s in subconds:
    for cond in ordered_conditions:
        values = data_by_condition[cond][s]
        mean = np.mean(values) if values else 0
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci = sem * stats.t.ppf(0.975, df=len(values)-1) if len(values) > 1 else 0
        means[s].append(mean)
        cis[s].append(ci)

# ======================
# Plotting
# ======================
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.25
x = np.arange(len(ordered_conditions))

colors = {
    "far": "#999999",
    "close": "#333333",
    "overall": "#cccccc"
}

for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width
    xpos = x + offset
    abbr = {"far": "F", "close": "C", "overall": "O"}
    bars = ax.bar(
        xpos, means[sub], width=bar_width, label=abbr[sub],
        color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none'
    )
    
    # Add number labels above bars
    for bar, mean in zip(bars, means[sub]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * ax.get_ylim()[1],
            f"{mean:.2f}",
            ha='center', va='bottom', fontsize=18
        )

ax.set_xticks(x)
ax.set_xticklabels([label_map[c] for c in ordered_conditions])
ax.set_ylabel("System Informativeness")
ax.set_ylim(0, max(max(means[sub]) + max(cis[sub]) for sub in subconds) * 1.1)
ax.grid(False)
sns.despine(ax=ax)

from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
title_handle = Patch(color='none', label='Context Type')
handles = [title_handle] + handles
labels = ['Test Context'] + labels

ax.legend(handles, labels,
          loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, frameon=False, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.5, fontsize=30)

plt.tight_layout()
plt.savefig("system_informativeness_upsample.pdf", dpi=300)
plt.show()
