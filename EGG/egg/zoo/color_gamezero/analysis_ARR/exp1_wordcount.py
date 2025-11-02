import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.patches import Patch

# Set aesthetic style
sns.set(style="white", font_scale=1.6, rc={
    'axes.labelweight': 'normal',
    'axes.titlesize': 18,
    'axes.labelsize': 45,
    'legend.title_fontsize': 45,
    'legend.fontsize': 45,
    'xtick.labelsize': 45,
    'ytick.labelsize': 45,
    'axes.titleweight': 'normal'
})

# Define conditions and custom x-axis order
# conditions = {
#     ("dump", "epoch0"): "SL w/o",
#     ("dump_context", "epoch0"): "SL w/",
#     ("dump", "epoch30"): "SL w/o + RL w/o",
#     ("dump_exp", "epoch30"): "SL w/o + RL w/",
#     ("dump_context", "epoch30"): "SL w/ + RL w/"
# }

conditions = {
    ("dump", "epoch0"):  r"$\mathrm{SL{-}}$",
    ("dump_context", "epoch0"): r"$\mathrm{SL{+}}$",
    ("dump", "epoch30"):  r"$\mathrm{SL{-}SL{-}}$",
    ("dump_exp", "epoch30"): r"$\mathrm{SL{-}SL{+}}$",
    ("dump_context", "epoch30"): r"$\mathrm{SL{+}SL{+}}$"
}


# ordered_labels = ["SL w/o", "SL w/", "SL w/o + RL w/o", "SL w/o + RL w/", "SL w/ + RL w/"]
ordered_labels = [
    r"$\mathrm{SL{-}}$",
    r"$\mathrm{SL{+}}$",
    r"$\mathrm{SL{-}SL{-}}$",
    r"$\mathrm{SL{-}SL{+}}$",
    r"$\mathrm{SL{+}SL{+}}$"
]

ordered_keys = [key for key, label in conditions.items() if label in ordered_labels]
ordered_keys = sorted(ordered_keys, key=lambda k: ordered_labels.index(conditions[k]))

# Define subconditions and colors
subconds = ["overall", "far", "close"]
subcond_colors = {
    "far": "#999999",  # medium gray
    "close": "#333333", # dark charcoal
    "overall": "#cccccc" # light gray
    # "overall": "#f3b8ad",  # soft red
    # "far": "#d4d4d4",      # soft gray
    # "close": "#95a3c5",    # muted blue
}
abbr = {"far": "F", "close": "C", "overall": "O"}
seed_ids = [111, 222, 333, 444, 555, 666, 777, 888, 999, 123, 234, 345, 456, 567, 678, 789, 891, 912]

# Collect data
data = {key: {sub: [] for sub in subconds} for key in conditions}
base_path = "../condition3_generated_ARR/experiment1/"

for (cond_key, epoch), label in conditions.items():
    for seed in seed_ids:
        # seed_dir = f"{cond_key}/msg_rf_seed{seed}"
        seed_dir = os.path.join(base_path, cond_key, f"msg_rf_seed{seed}")
        for subcond in subconds:
            filename = f"{epoch}_word_used_{subcond}.csv"
            file_path = os.path.join(seed_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                count = (df["count"] > 0).sum()
                data[(cond_key, epoch)][subcond].append(count)
            else:
                print(f"Missing file: {file_path}")

# Compute means and confidence intervals
means = {s: [] for s in subconds}
cis = {s: [] for s in subconds}
for s in subconds:
    for key in ordered_keys:
        values = data[key][s]
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci = sem * stats.t.ppf(0.975, df=len(values)-1) if len(values) > 1 else 0
        means[s].append(mean)
        cis[s].append(ci)

print("\nMean ± CI for each condition:")
for i, key in enumerate(ordered_keys):
    label = conditions[key]
    print(f"\nCondition: {label}")
    for s in subconds:
        mean_val = means[s][i]
        ci_val = cis[s][i]
        print(f"  {s.capitalize():<7}: {mean_val:.2f} ± {ci_val:.2f}")

# Plotting
fig, ax = plt.subplots(figsize=(20, 10))
bar_width = 0.22
x = np.arange(len(ordered_keys))

for i, sub in enumerate(subconds):
    offset = (i - 1) * bar_width
    xpos = x + offset
    bars = ax.bar(xpos, means[sub], width=bar_width, label=abbr[sub],
                  color=subcond_colors[sub], yerr=cis[sub], capsize=4, edgecolor='none')


# X/Y axis setup
ax.set_xticks(x)
# ax.set_xticklabels([
#     "SL w/o",
#     "SL w/",
#     "SL w/o \n+ RL w/o",
#     "SL w/o \n+ RL w/",
#     "SL w/ \n+ RL w/"
# ])

ax.set_xticklabels(ordered_labels)
# ax.set_xticklabels([conditions[k] for k in ordered_keys], rotation=0, ha='right')
ax.set_ylabel("Number of Lexical Items")
ax.set_ylim(0, max(max(means[s][i] + cis[s][i] for i in range(len(x))) for s in subconds) + 5)
ax.grid(False)
sns.despine(ax=ax)

# Create custom legend with title
handles, labels = ax.get_legend_handles_labels()
title_handle = Patch(color='none', label='Test Context')
handles = [title_handle] + handles
labels = ['Test Context'] + labels

ax.legend(handles, labels,
          loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, frameon=False, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.5, fontsize=45)

plt.tight_layout()
plt.savefig("word_type_usage_parta.pdf", dpi=300)
plt.show()


