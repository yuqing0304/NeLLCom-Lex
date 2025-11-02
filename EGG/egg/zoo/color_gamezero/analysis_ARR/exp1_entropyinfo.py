import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


base_path = "../condition3_generated_ARR/experiment1/"

# Set modern ggplot-style aesthetics
sns.set(style="white", font_scale=1.6, rc={
    'axes.labelweight': 'normal',
    'axes.titlesize': 18,
    'axes.labelsize': 36,
    'legend.title_fontsize': 36,
    'legend.fontsize': 36,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'axes.titleweight': 'normal'
})

# Define condition mappings
conditions = {
    ("dump", "epoch0"): "SL w/o",
    ("dump_context", "epoch0"): "SL w/",
    ("dump", "epoch30"): "SL w/o + RL w/o",
    ("dump_exp", "epoch30"): "SL w/o + RL w/",
    ("dump_context", "epoch30"): "SL w/ + RL w/"
}

# Entropy over word types in the human data: 3.8340

# # Improved x-axis label formatting with line breaks
# label_map = {
#     "SL w/o": "SL w/o",
#     "SL w/": "SL w/",
#     "SL w/o + RL w/o": "SL w/o\n+ RL w/o",
#     "SL w/o + RL w/": "SL w/o\n+ RL w/",
#     "SL w/ + RL w/": "SL w/\n+ RL w/"
# }

label_map = {
    "SL w/o": r"$\mathrm{SL{-}}$",
    "SL w/":  r"$\mathrm{SL{+}}$",
    "SL w/o + RL w/o": r"$\mathrm{SL{-}SL{-}}$",
    "SL w/o + RL w/":  r"$\mathrm{SL{-}SL{+}}$",
    "SL w/ + RL w/": r"$\mathrm{SL{+}SL{+}}$"
}

# label_renames = {
#     "SL w/o context (zeroed) + RL w/o context": r"$\mathrm{SL{-}}$",
#     # "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}}$",
#     "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}}$"
#     # "SL w/o context (zeroed) + RL w/ context":  r"$\mathrm{SL{-}}}$",
# }

ordered_labels = list(label_map.values())


# Custom 5-color palette (including the 3 colors you provided)
# custom_palette = [
#     "#f3b8ad",  # soft red
#     "#d4d4d4",  # soft gray
#     "#95a3c5",  # muted blue
#     "#d0e3e7",  # soft cyan
#     "#e4c9b3"   # soft beige
# ]

custom_palette = [
    "#e0e0e0",  # very light gray
    "#c0c0c0",  # light gray
    "#a0a0a0",  # medium gray
    "#707070",  # dark gray
    "#404040"   # very dark gray
]


# Collect entropy data
entropy_records = []

for (folder, epoch), label in conditions.items():
    # cond_path = os.path.join(".", folder)
    cond_path = os.path.join(base_path, folder)  


    for seed_folder in os.listdir(cond_path):
        if re.match(r"msg_rf_seed\d+$", seed_folder):
            file_path = os.path.join(cond_path, seed_folder, f"{epoch}_informativeness_overall.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'informativeness' in df.columns and not df['informativeness'].isnull().all():
                    norm_info = df["informativeness"] / df["informativeness"].sum()
                    entropy = -np.sum(norm_info * np.log(norm_info + 1e-12))
                    entropy_records.append({
                        "condition": label_map[label],
                        "entropy": entropy
                    })
                else:
                    print(f"Skipping (missing/invalid informativeness): {file_path}")
            else:
                print(f"Missing file: {file_path}")

# Convert to DataFrame
entropy_df = pd.DataFrame(entropy_records)

# Ensure ordering
entropy_df["condition"] = pd.Categorical(entropy_df["condition"], categories=ordered_labels, ordered=True)

# Plot
# plt.figure(figsize=(20, 10))
# ax = sns.barplot(
#     data=entropy_df,
#     x="condition",
#     y="entropy",
#     hue="condition",  # Required for palette mapping
#     errorbar=("ci", 95),
#     palette=custom_palette,
#     legend=False,
#     width=0.5 
# )
plt.figure(figsize=(15, 10))
ax = sns.barplot(
    data=entropy_df,
    x="condition",
    y="entropy",
    hue="condition",  # Required for palette mapping
    errorbar=("ci", 95),
    palette=custom_palette,
    legend=False,
    width=0.45  # Thinner bars
)

# Styling
ax.set_ylabel("Entropy of Informativeness")
ax.set_xlabel("")
ax.set_title("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
sns.despine()

plt.xticks(rotation=0)

# Add numbers above the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2., height + 0.08,  # Position the number slightly above the bar
        f'{height:.2f}',  # Format to 2 decimal places
        ha='center', 
        va='bottom', 
        fontsize=36
    )

# Add horizontal line for human data entropy
human_entropy = 3.8340
ax.axhline(y=human_entropy, color="black", linestyle="--", linewidth=2)

# Add label above the line
ax.text(
    x=len(ordered_labels) - 0.5,  # Right side of plot
    y=human_entropy + 0.08,
    s=f"Human: {human_entropy:.2f}",
    ha="right",
    va="bottom",
    fontsize=36,
    color="black"
)

ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig("entropy_plotparta.pdf", dpi=300, bbox_inches='tight')
plt.show()








# import os
# import pandas as pd
# import numpy as np

# import re  # for matching seed folder names

# # Top-level condition folders
# conditions = ["dump", "dump_context", "dump_exp"]
# entropies_by_condition = {}

# for cond in conditions:
#     cond_path = os.path.join(".", cond)
#     entropy_list = []

#     for folder_name in os.listdir(cond_path):
#         # Only process folders that match the seed pattern
#         if re.match(r"msg_rf_seed\d+$", folder_name):
#             seed_path = os.path.join(cond_path, folder_name)
#             file_path = os.path.join(seed_path, "epoch30_informativeness_overall.csv")

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





# import pandas as pd
# import numpy as np
# import os

# entropies = {}


# for f in os.listdir():
#     if 'dump' in f:
#         path = f + '/'
#         e = []
#         for file in os.listdir(path):
#             file_path = path + file
#             data = pd.read_csv(file_path)
#             norm_info = data['informativeness'] / data['informativeness'].sum()
#             entropy = -np.sum(norm_info * np.log(norm_info + 1e-12))
#             e.append(entropy)
#         entropies[path] = np.mean(e)

# print(entropies)