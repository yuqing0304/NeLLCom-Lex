import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==== Setup ====

BASE_PATH = "../condition3_generated_ARR/experiment2/"

conditions = {
    "before_RL": os.path.join(BASE_PATH, "condition_a/dump_context"),  # Pre-RL
    "a": os.path.join(BASE_PATH, "condition_a/dump_context"),          # far:close 0:100
    "b": os.path.join(BASE_PATH, "condition_b/dump_context"),          # far:close 50:50
    "c": os.path.join(BASE_PATH, "condition_c/dump_context"),          # far:close 100:0
}

label_map = {
    "before_RL": "Before RL",
    "a": "AllClose",
    "b": "HalfHalf",
    "c": "AllFar",
}

ordered_conditions = list(label_map.keys())
subconds = ["overall", "far", "close"]
seeds = [f"msg_rf_seed{i}" for i in [111, 222, 333, 444, 555, 666, 777, 888, 999, 123, 234, 345, 456, 567, 678, 789, 891, 912]]

files_for_condition = {
    "before_RL": {
        "overall": "epoch0_informativeness_overall.csv",
        "far": "epoch0_far_informativeness.csv",
        "close": "epoch0_close_informativeness.csv"
    },
    "default": {
        "overall": "epoch30_informativeness_overall.csv",
        "far": "epoch30_far_informativeness.csv",
        "close": "epoch30_close_informativeness.csv"
    }
}

# ==== Load Entropy Data ====
entropy_data = {cond: {s: [] for s in subconds} for cond in conditions}

for cond, base_path in conditions.items():
    file_map = files_for_condition["before_RL"] if cond == "before_RL" else files_for_condition["default"]
    for sub in subconds:
        file_name = file_map[sub]
        for seed in seeds:
            file_path = os.path.join(base_path, seed, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.dropna(subset=["informativeness"])
                if not df.empty:
                    probs = df["informativeness"] / df["informativeness"].sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-12))
                    entropy_data[cond][sub].append(entropy)
            else:
                print(f"Missing: {file_path}")

# ==== Compute Stats ====
means, cis = {s: [] for s in subconds}, {s: [] for s in subconds}
for s in subconds:
    for cond in ordered_conditions:
        values = entropy_data[cond][s]
        if values:
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / np.sqrt(len(values))
            ci = sem * stats.t.ppf(0.975, df=len(values)-1)
        else:
            mean, ci = np.nan, np.nan
        means[s].append(mean)
        cis[s].append(ci)

# ==== Plot ====
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

fig, ax = plt.subplots(figsize=(13, 6))
bar_width = 0.22
x = np.arange(len(ordered_conditions))

# colors = {
#     "far": "#d4d4d4",
#     "close": "#95a3c5",
#     "overall": "#f3b8ad"
# }

colors = {
    "far": "#999999",  # medium gray
    "close": "#333333", # dark charcoal
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

ax.set_xticks(x)
ax.set_xticklabels([label_map[c] for c in ordered_conditions])
ax.set_ylabel("Entropy of Informativeness")
ax.set_ylim(0, max(max(m + c if not np.isnan(c) else m for m, c in zip(means[sub], cis[sub])) for sub in subconds) + 0.5)
ax.grid(False)
sns.despine(ax=ax)

# Legend with title
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
title_handle = Patch(color='none', label='Test Context')
handles = [title_handle] + handles
labels = ['Test Context'] + labels
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, frameon=False, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.5, fontsize=30)

plt.tight_layout()
plt.savefig("entropy_plotpartb.pdf", dpi=300)
plt.show()





# import os
# import pandas as pd
# import numpy as np
# import re  # for matching seed folder names

# # Nested condition folders
# conditions = ["condition_a/dump_context", "condition_b/dump_context", "condition_c/dump_context"]
# entropies_by_condition = {}

# for cond in conditions:
#     cond_path = os.path.join(".", cond)
#     entropy_list = []

#     for folder_name in os.listdir(cond_path):
#         # Only process folders that match the seed pattern
#         if re.match(r"msg_rf_seed\d+$", folder_name):
#             seed_path = os.path.join(cond_path, folder_name)
#             file_path = os.path.join(seed_path, "close_informativeness.csv") # epoch30_informativeness_overall.csv

#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path)
#                 if 'informativeness' in df.columns and not df['informativeness'].isnull().all():
#                     norm_info = df["informativeness"] / df["informativeness"].sum()
#                     entropy = -np.sum(norm_info * np.log(norm_info + 1e-12))
#                     entropy_list.append(entropy)
#                 else:
#                     print(f"Skipping (missing/invalid informativeness): {file_path}")
#             else:
#                 print(f"Missing file: {file_path}")

#     if entropy_list:
#         entropies_by_condition[cond] = {
#             "mean_entropy": np.mean(entropy_list),
#             "all_entropies": entropy_list,
#             "n_seeds": len(entropy_list)
#         }

# # Print results with mean, std, and standard error
# for cond, stats in entropies_by_condition.items():
#     entropies = np.array(stats["all_entropies"])
#     mean_entropy = np.mean(entropies)
#     std_entropy = np.std(entropies, ddof=1)  # sample std dev
#     se_entropy = std_entropy / np.sqrt(len(entropies))

#     print(f"\nCondition: {cond}")
#     print(f"  Mean entropy: {mean_entropy:.4f}")
#     print(f"  Std entropy: {std_entropy:.4f}")
#     print(f"  SE entropy: {se_entropy:.4f}")
#     print(f"  # Seeds: {len(entropies)}")
