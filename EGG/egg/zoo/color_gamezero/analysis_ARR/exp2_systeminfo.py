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
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'axes.titleweight': 'normal'
})

# Define conditions relative to a base path
BASE_PATH = "../condition3_generated_ARR/experiment2/"


# File mapping
file_map = {
    "before_RL": "./condition_b/dump_context/spk_informativeness.txt",
    "a": "./condition_a/dump_context/informativeness.txt",
    "b": "./condition_b/dump_context/informativeness.txt",
    "c": "./condition_c/dump_context/informativeness.txt",
}

file_map = {
    "before_RL": os.path.join(BASE_PATH, "condition_a/dump_context/spk_informativeness.txt"),
    "a": os.path.join(BASE_PATH, "condition_a/dump_context/informativeness.txt"),
    "b": os.path.join(BASE_PATH, "condition_b/dump_context/informativeness.txt"),
    "c": os.path.join(BASE_PATH, "condition_c/dump_context/informativeness.txt"),
}   


label_map = {
    "before_RL": "Before RL",
    "a": "AllClose",
    "b": "HalfHalf",
    "c": "AllFar",
}

ordered_conditions = list(label_map.keys())
subconds = ["overall", "far", "close"]

# Initialize data storage
data_by_condition = {cond: {sub: [] for sub in subconds} for cond in file_map}

# Parse files and extract informativeness scores
for cond, filename in file_map.items():
    with open(filename, "r") as f:
        for line in f:
            if "Overall Lexical System Informativeness" in line:
                data_by_condition[cond]["overall"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
            elif "far" in line:
                data_by_condition[cond]["far"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
            elif "close" in line:
                data_by_condition[cond]["close"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))

# Compute means and confidence intervals (95% CI)
means, cis = {s: [] for s in subconds}, {s: [] for s in subconds}
for s in subconds:
    for cond in ordered_conditions:
        values = data_by_condition[cond][s]
        mean = np.mean(values) if values else 0
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci = sem * stats.t.ppf(0.975, df=len(values)-1) if len(values) > 1 else 0
        means[s].append(mean)
        cis[s].append(ci)

fig, ax = plt.subplots(figsize=(13, 6))

bar_width = 0.22
x = np.arange(len(ordered_conditions))

colors = {
    "far": "#999999",    # medium gray
    "close": "#333333",  # dark charcoal
    "overall": "#cccccc" # light gray
}

# Plot bars for each subcondition with offset, color, and CI error bars
for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width  # shifts bars left/center/right
    xpos = x + offset
    abbr = {"far": "F", "close": "C", "overall": "O"}
    bars = ax.bar(xpos, means[sub], width=bar_width, label=abbr[sub],
                  color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none')

    # Optionally label bars with values:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}",
                ha='center', va='bottom', fontsize=18)

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
plt.savefig("system_informativeness_partb.pdf", dpi=300)
plt.show()
