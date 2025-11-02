import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

fig, (ax, ax_legend) = plt.subplots(ncols=2, figsize=(16, 7), gridspec_kw={'width_ratios': [4.5, 1]})

sns.set(style="white", font_scale=1.6, rc={
    'axes.labelweight': 'normal',
    'axes.titlesize': 18,
    'axes.labelsize': 30,   # reduced from 35
    'legend.title_fontsize': 30,  # reduced for better fit
    'legend.fontsize': 30,        # reduced for better fit
    'xtick.labelsize': 30,        # reduced slightly
    'ytick.labelsize': 30,
    'axes.titleweight': 'normal'
})

# # Conditions and setup
# conditions = {
#     "before_RL": "condition_a/",  # Pre-RL
#     "a": "condition_a/",          # far:close 0:100
#     "b": "condition_b/",          # far:close 50:50
#     "c": "condition_c/",          # far:close 100:0
# }


# Define conditions relative to a base path
BASE_PATH = "../condition3_generated_ARR/experiment2/"

conditions = {
    "before_RL": os.path.join(BASE_PATH, "condition_a/"),  # Pre-RL
    "a": os.path.join(BASE_PATH, "condition_a/"),          # far:close 0:100
    "b": os.path.join(BASE_PATH, "condition_b/"),          # far:close 50:50
    "c": os.path.join(BASE_PATH, "condition_c/"),          # far:close 100:0
}


label_map = {
    "before_RL": "Before RL",
    "a": "AllClose",
    "b": "HalfHalf",
    "c": "AllFar",
}

ordered_conditions = list(label_map.keys())
subconds = ["overall", "far", "close"]
seeds = [f"dump_context/msg_rf_seed{i}" for i in [111, 222, 333, 444, 555, 666, 777, 888, 999, 123, 234, 345, 456, 567, 678, 789, 891, 912]]

# Initialize data storage
data_by_condition = {cond: {sub: [] for sub in subconds} for cond in conditions}

# Load data
for cond, path in conditions.items():
    for subcond in subconds:
        for seed in seeds:
            epoch = "epoch0" if cond == "before_RL" else "epoch30"
            file_path = os.path.join(path, seed, f"{epoch}_word_used_{subcond}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                count = (df["count"] > 0).sum()
                data_by_condition[cond][subcond].append(count)
            else:
                print(f"Missing file: {file_path}")

# Compute means and confidence intervals
means, cis = {s: [] for s in subconds}, {s: [] for s in subconds}
for s in subconds:
    for cond in ordered_conditions:
        values = data_by_condition[cond][s]
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci = sem * stats.t.ppf(0.975, df=len(values)-1) if len(values) > 1 else 0
        means[s].append(mean)
        cis[s].append(ci)

fig, ax = plt.subplots(figsize=(13, 6))  # removed ax_legend
# Plotting with space for vertical legend
# fig, (ax, ax_legend) = plt.subplots(ncols=2, figsize=(13, 6), gridspec_kw={'width_ratios': [4, 1]})

bar_width = 0.22
x = np.arange(len(ordered_conditions))



colors = {
    "far": "#999999",  # medium gray
    "close": "#333333", # dark charcoal
    "overall": "#cccccc" # light gray
}

# colors = {
#     "far": "#b0b0b0",     #"#d4d4d4",     
#     "close": "#3b4c6b",   #"#95a3c5", 
#     "overall": "#f2a59b"    # "#f3b8ad"  
# }

for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width
    xpos = x + offset
    abbr = {"far": "F", "close": "C", "overall": "O"}
    bars = ax.bar(xpos, means[sub], width=bar_width, label=abbr[sub],
                color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}",
                ha='center', va='bottom', fontsize=22, rotation=15)  
# Main plot labels
ax.set_xticks(x)
ax.set_xticklabels([label_map[c] for c in ordered_conditions])
ax.set_ylabel("Number of Word Types")
# ax.set_xlabel("Training Condition")
ax.set_ylim(0, max(max(means[sub]) + max(cis[sub]) for sub in subconds) + 5)
ax.grid(False)
sns.despine(ax=ax)


from matplotlib.patches import Patch

# Get actual legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Create a dummy (invisible) handle for the title
title_handle = Patch(color='none', label='Test Context')

# Prepend the dummy title to the handles and labels
handles = [title_handle] + handles
labels = ['Test Context'] + labels

# Create the legend with 4 columns: title + 3 entries
ax.legend(handles, labels,
          loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, frameon=False, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.5, fontsize=30)




plt.tight_layout()
plt.savefig("word_type_usage_partb.pdf", dpi=300)
plt.show()
