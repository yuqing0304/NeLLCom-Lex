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
    'xtick.labelsize': 20,
    'ytick.labelsize': 30,
    'axes.titleweight': 'normal'
})

# ======================
# Conditions and setup
# ======================
BASE_PATH = "../../color_game_group/"
total_epochs = 30

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

# Epochs per condition (if same for all)
epoch_map = {
    "b0": total_epochs // 1,
    "b100": total_epochs // 1,
    "b200": total_epochs // 1,
}

ordered_conditions = ["b0", "b100", "b200"]
subconds = ["overall", "far", "close"]

# Seeds
seeds = [f"dump_context/msg_rf_seed{i}" for i in [
    111, 123, 333, 345, 444, 555, 567, 666, 777, 912
]]

# ======================
# Data loading
# ======================
data_by_condition = {cond: {sub: [] for sub in subconds} for cond in conditions}

for cond, path in conditions.items():
    epoch = epoch_map[cond]
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
fig, ax = plt.subplots(figsize=(12, 6))  # slightly wider for 3 conditions

bar_width = 0.25
x = np.arange(len(ordered_conditions))

colors = {
    "far": "#999999",    # medium gray
    "close": "#333333",  # dark charcoal
    "overall": "#cccccc" # light gray
}

for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width
    xpos = x + offset
    abbr = {"far": "F", "close": "C", "overall": "O"}
    bars = ax.bar(
        xpos, means[sub], width=bar_width, label=abbr[sub],
        color=colors[sub], yerr=cis[sub], capsize=4, edgecolor='none'
    )

    # Add numeric labels
    for bar, mean in zip(bars, means[sub]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * ax.get_ylim()[1],
            f"{mean:.1f}",
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
plt.savefig("word_type_usage_upsample.pdf", dpi=300)
plt.show()
