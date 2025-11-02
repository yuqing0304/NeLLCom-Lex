import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
import re
from scipy import stats  # added for CI computation

# Set seaborn theme to resemble ggplot2's theme_minimal()
sns.set(style="white", context="talk", font_scale=1.2)

plt.rcParams.update({
    'grid.color': '#e5e5e5',
    'grid.linewidth': 0.5,
    'axes.titlesize': 60,
    'axes.titleweight': 'bold',
    'axes.labelsize': 60,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'figure.dpi': 300,
    'figure.figsize': (10, 12),
    'axes.titlepad': 20,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0,
})

def read_and_plot_accuracy(log_folder, log_prefix, label, accuracy_type):
    log_files = [f for f in os.listdir(log_folder) if f.startswith(log_prefix) and f.endswith('.txt')]
    print(f"Reading from {log_folder}: {log_files}")

    all_epochs = []
    all_valid_accs = []
    all_far_accs = []
    all_close_accs = []
    all_split_accs = []

    for log_file in log_files:
        log_path = os.path.join(log_folder, log_file)
        epochs = []
        valid_accs = []
        far_accs = []
        close_accs = []
        split_accs = []

        with open(log_path, 'r') as f:
            for line in f:
                epoch_match = re.match(r"Epoch (\d+), valid acc: ([0-9.]+), far_acc: ([0-9.]+), close_acc: ([0-9.]+), split_acc: ([0-9.]+)", line)
                if epoch_match:
                    epoch = int(epoch_match.group(1)) + 1
                    valid_acc = float(epoch_match.group(2))
                    far_acc = float(epoch_match.group(3))
                    close_acc = float(epoch_match.group(4))
                    split_acc = float(epoch_match.group(5))

                    epochs.append(epoch)
                    valid_accs.append(valid_acc)
                    far_accs.append(far_acc)
                    close_accs.append(close_acc)
                    split_accs.append(split_acc)

        epochs = np.array(epochs, dtype=int)
        valid_accs = np.array(valid_accs, dtype=float)
        far_accs = np.array(far_accs, dtype=float)
        close_accs = np.array(close_accs, dtype=float)
        split_accs = np.array(split_accs, dtype=float)

        all_epochs.append(epochs)
        all_valid_accs.append(valid_accs)
        all_far_accs.append(far_accs)
        all_close_accs.append(close_accs)
        all_split_accs.append(split_accs)

    all_epochs = np.concatenate(all_epochs)
    all_valid_accs = np.concatenate(all_valid_accs)
    all_far_accs = np.concatenate(all_far_accs)
    all_close_accs = np.concatenate(all_close_accs)
    all_split_accs = np.concatenate(all_split_accs)

    unique_epochs = np.unique(all_epochs)

    def calculate_avg_ci(accs):
        avg_accs = []
        ci_halfwidths = []
        for epoch in unique_epochs:
            epoch_indices = all_epochs == epoch
            values = accs[epoch_indices]
            n = len(values)
            avg_acc = np.mean(values)
            std_err = np.std(values, ddof=1) / np.sqrt(n)
            ci_halfwidth = stats.t.ppf(0.975, df=n-1) * std_err if n > 1 else 0
            avg_accs.append(avg_acc)
            ci_halfwidths.append(ci_halfwidth)
        return np.array(avg_accs), np.array(ci_halfwidths)

    avg_valid_accs, ci_valid_accs = calculate_avg_ci(all_valid_accs)
    avg_far_accs, ci_far_accs = calculate_avg_ci(all_far_accs)
    avg_close_accs, ci_close_accs = calculate_avg_ci(all_close_accs)
    avg_split_accs, ci_split_accs = calculate_avg_ci(all_split_accs)

    if accuracy_type == 'valid':
        avg_accs = avg_valid_accs
        ci_accs = ci_valid_accs
    elif accuracy_type == 'far':
        avg_accs = avg_far_accs
        ci_accs = ci_far_accs
    elif accuracy_type == 'close':
        avg_accs = avg_close_accs
        ci_accs = ci_close_accs
    elif accuracy_type == 'split':
        avg_accs = avg_split_accs
        ci_accs = ci_split_accs
    else:
        raise ValueError(f"Unknown accuracy type: {accuracy_type}")

    if 30 in unique_epochs:
        idx_30 = np.where(unique_epochs == 30)[0][0]
        acc_30 = avg_accs[idx_30]
        ci_30 = ci_accs[idx_30]
        print(f"{label}: Accuracy at epoch 30 = {acc_30:.3f} ± {ci_30:.3f}")
    else:
        print(f"{label}: Epoch 30 not found in logs.")



#### spk
    colors = {
        "SL w/o context (zeroed) + RL w/o context": "#b0b0b0",  
        "SL w/ context + RL w/ context": "#3b4c6b",            
        # "SL w/o context (zeroed) + RL w/ context": "#f2a59b"     
    }

    # label_renames = {
    #     "SL w/o context (zeroed) + RL w/o context": r"$\mathrm{SL{-}RL{-}}$",
    #     "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}RL{+}}$",
    #     # "SL w/o context (zeroed) + RL w/ context":  r"$\mathrm{SL{-}RL{+}}$",
    # }

    label_renames = {
        "SL w/o context (zeroed) + RL w/o context": r"$\mathrm{SL{-}}$",
        # "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}}$",
        "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}}$"
        # "SL w/o context (zeroed) + RL w/ context":  r"$\mathrm{SL{-}}}$",
    }



# #### lst
#     colors = {
#         "SL w/ context + RL w/ context": "#3b4c6b",            
#     }


#     label_renames = {
#         # "SL w/o context (zeroed) + RL w/o context": r"$\mathrm{SL{-}}$",
#         # "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}}$",
#         "SL w/ context + RL w/ context":            r"$\mathrm{SL}$"
#         # "SL w/o context (zeroed) + RL w/ context":  r"$\mathrm{SL{-}}}$",
#     }

    renamed_label = label_renames.get(label, label)
    plt.plot(unique_epochs, avg_accs, label=renamed_label, color=colors[label], linewidth=7, markeredgewidth=0)
    plt.fill_between(unique_epochs, avg_accs - ci_accs, avg_accs + ci_accs, color=colors[label], alpha=0.2)



def main():
    parser = argparse.ArgumentParser(description="Plot validation accuracy from training logs.")
    parser.add_argument('--log_folder1', type=str, required=True, help="First folder containing log files")
    parser.add_argument('--log_prefix1', type=str, required=True, help="Prefix of the first log files")
    parser.add_argument('--log_folder2', type=str, required=True, help="Second folder containing log files")
    parser.add_argument('--log_prefix2', type=str, required=True, help="Prefix of the second log files")
    parser.add_argument('--log_folder3', type=str, required=True, help="Third folder containing log files")
    parser.add_argument('--log_prefix3', type=str, required=True, help="Prefix of the third log files")
    parser.add_argument('--accuracy_type', type=str, choices=['valid', 'far', 'close', 'split'], required=True, help="Accuracy type to plot (valid, far, close, split)")
    parser.add_argument('--output_plot', type=str, required=True, help="Output plot file name")

    args = parser.parse_args()

    read_and_plot_accuracy(args.log_folder1, args.log_prefix1, "SL w/o context (zeroed) + RL w/o context", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder2, args.log_prefix2, "SL w/ context + RL w/ context", args.accuracy_type)
    # read_and_plot_accuracy(args.log_folder3, args.log_prefix3, "SL w/o context (zeroed) + RL w/ context", args.accuracy_type)

    # plt.xlabel("Epoch", labelpad=10)
    plt.ylabel("Spk Acc", labelpad=10)
    plt.xticks(np.arange(0, 31, 10))
    plt.ylim(0, 1)
    plt.legend(loc='lower right', frameon=False, fontsize=60)
    plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.gca().set_yticklabels([])  # removes y-axis tick labels
    plt.tight_layout()

    parent_output_dir = os.path.abspath(os.path.join(args.log_folder1, os.pardir))
    output_path = os.path.join(parent_output_dir, args.output_plot)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
