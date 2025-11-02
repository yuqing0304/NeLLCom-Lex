import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

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

# ======================
# Conditions and setup
# ======================
BASE_PATH = "../../color_game_group/"

total_epochs = 30
conditions = {
    "b1": os.path.join(BASE_PATH, "condition_b_1lst/"),
    "b2": os.path.join(BASE_PATH, "condition_b_2lst/"),
    "b6": os.path.join(BASE_PATH, "condition_b_6lst/"),
    "b10": os.path.join(BASE_PATH, "condition_b_10lst/"),
}

label_map = {
    "b1": "1 lst",
    "b2": "2 lsts",
    "b6": "6 lsts",
    "b10": "10 lsts",
}

# Epochs per condition
# epoch_map = {
#     "b1": 30,
#     "b2": 15,
#     "b6": 5,
#     "b10": 3,
# }

epoch_map = {
    "b1": total_epochs // 1,
    "b2": total_epochs // 2,
    "b6": total_epochs // 6,
    "b10": total_epochs // 10,
}

ordered_conditions = list(label_map.keys())
subconds = ["overall", "far", "close"]

# Seeds
seeds = [f"dump_context/msg_rf_seed{i}" for i in [
    111, 123, 222, 333, 345, 456, 567, 777, 891, 999
]]

# ======================
# Data loading
# ======================
data_by_condition = {cond: {sub: [] for sub in subconds} for cond in conditions}

for cond, path in conditions.items():
    epoch = epoch_map[cond]  # pick the right epoch number
    for subcond in subconds:
        for seed in seeds:
            file_path = os.path.join(path, seed, f"epoch{epoch}_word_used_{subcond}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                count = (df["count"] > 0).sum()
                data_by_condition[cond][subcond].append(count)
            else:
                print(f"Missing file: {file_path}")


# ======================
# Compute means + CI
# ======================
means, cis = {s: [] for s in subconds}, {s: [] for s in subconds}
for s in subconds:
    for cond in ordered_conditions:
        values = data_by_condition[cond][s]
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci = sem * stats.t.ppf(0.975, df=len(values)-1) if len(values) > 1 else 0
        means[s].append(mean)
        cis[s].append(ci)

# ======================
# Plotting
# ======================
fig, ax = plt.subplots(figsize=(13, 6))

bar_width = 0.22
x = np.arange(len(ordered_conditions))

colors = {
    "far": "#999999",   # medium gray
    "close": "#333333", # dark charcoal
    "overall": "#cccccc" # light gray
}

# for i, sub in enumerate(subconds):
#     offset = (i - 1) * bar_width
#     xpos = x + offset
#     abbr = {"far": "F", "close": "C", "overall": "O"}
#     bars = ax.bar(xpos, means[sub], width=bar_width, label=abbr[sub],
#                   color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none')

for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width
    xpos = x + offset
    abbr = {"far": "F", "close": "C", "overall": "O"}
    bars = ax.bar(
        xpos, means[sub], width=bar_width, label=abbr[sub],
        color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none'
    )

    # Add number above each bar
    for bar, mean in zip(bars, means[sub]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # center of bar
            height + 0.01 * ax.get_ylim()[1],   # slightly above bar
            f"{mean:.1f}",                      # format (1 decimal)
            ha="center", va="bottom", fontsize=18
        )


ax.set_xticks(x)
ax.set_xticklabels([label_map[c] for c in ordered_conditions])
ax.set_ylabel("Number of Word Types")
ax.set_ylim(0, max(max(means[sub]) + max(cis[sub]) for sub in subconds) + 5)
ax.grid(False)
sns.despine(ax=ax)

from matplotlib.patches import Patch

handles, labels = ax.get_legend_handles_labels()
title_handle = Patch(color='none', label='Test Context')
handles = [title_handle] + handles
labels = ['Test Context'] + labels

ax.legend(handles, labels,
          loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, frameon=False, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.5, fontsize=30)

plt.tight_layout()
plt.savefig("word_type_usage_group.pdf", dpi=300)
plt.show()
