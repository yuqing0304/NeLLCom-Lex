import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)


plt.rcParams.update({
    'grid.color': '#e5e5e5',
    'grid.linewidth': 0.5,
    'axes.titlesize': 60,
    'axes.titleweight': 'bold',
    'axes.labelsize': 60,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'figure.dpi': 300,
    'figure.figsize': (11, 13),
    'axes.titlepad': 20,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0,
})



def read_and_plot_accuracy(log_folder, log_prefix, label, accuracy_type):
    log_files = [f for f in os.listdir(log_folder) if f.startswith(log_prefix) and f.endswith('.txt')]
    print(f"Reading from {log_folder}: {log_files}")
    
    all_epochs = []
    all_accs = []

    for log_file in log_files:
        log_path = os.path.join(log_folder, log_file)
        epochs = []
        accs = []

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry.get("mode") == "test":
                        epoch = log_entry["epoch"]
                        acc = log_entry[accuracy_type]
                        epochs.append(epoch)
                        accs.append(acc)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in file: {log_file}")

        if epochs and accs:
            epochs = np.array(epochs, dtype=int)
            accs = np.array(accs, dtype=float)
            all_epochs.append(epochs)
            all_accs.append(accs)
    
    if not all_epochs or not all_accs:
        print(f"No valid data found in {log_folder} with prefix {log_prefix}.")
        return

    all_epochs = np.concatenate(all_epochs)
    all_accs = np.concatenate(all_accs)
    unique_epochs = np.unique(all_epochs)

    def calculate_avg_std(accs):
        avg_accs = []
        std_accs = []
        for epoch in unique_epochs:
            epoch_indices = all_epochs == epoch
            avg_acc = np.mean(accs[epoch_indices])
            std_acc = np.std(accs[epoch_indices])
            avg_accs.append(avg_acc)
            std_accs.append(std_acc)
        return np.array(avg_accs), np.array(std_accs)

    avg_accs, std_accs = calculate_avg_std(all_accs)

    # plt.plot(unique_epochs, avg_accs, label=label)
    # plt.fill_between(unique_epochs, avg_accs - std_accs, avg_accs + std_accs, alpha=0.3)
    # colors = {
    #     "far:close 0/100": "#fdae6b",       # orange
    #     "far:close 50/50": "#74c476",       # green
    #     "far:close 100/0": "#6baed6"        # blue
    # }

    colors = {
        "far:close 100/0": "#b0b0b0",  
        "far:close 0/100": "#3b4c6b",            
        "far:close 50/50": "#f2a59b"     
    }

    label_short = {
        "far:close 0/100": "All Close",
        "far:close 50/50": "Half Half",
        "far:close 100/0": "All Far"
    }

    color = colors.get(label, "black")
    label_text = label_short.get(label, label)

    plt.plot(unique_epochs, avg_accs, label=label_text, color=color, linewidth=7, markeredgewidth=0)
    plt.fill_between(unique_epochs, avg_accs - std_accs, avg_accs + std_accs, alpha=0.3, color=color)


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy from training logs.")
    parser.add_argument('--log_folder1', type=str, required=True)
    parser.add_argument('--log_prefix1', type=str, required=True)
    parser.add_argument('--log_folder2', type=str, required=True)
    parser.add_argument('--log_prefix2', type=str, required=True)
    parser.add_argument('--log_folder3', type=str, required=True)
    parser.add_argument('--log_prefix3', type=str, required=True)
    parser.add_argument('--accuracy_type', type=str, choices=['acc', 'acc_far', 'acc_close', 'acc_split'], required=True)
    parser.add_argument('--output_plot', type=str, required=True)

    args = parser.parse_args()

    read_and_plot_accuracy(args.log_folder1, args.log_prefix1, "far:close 0/100", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder2, args.log_prefix2, "far:close 50/50", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder3, args.log_prefix3, "far:close 100/0", args.accuracy_type)

    # plt.xlabel("Epoch")
    plt.ylabel("Overall Acc")
    # plt.title(f"Accuracy vs. Epoch for {args.accuracy_type.capitalize()} Accuracy")
    plt.xticks(np.arange(0, 31, 10))
    # plt.legend(loc='lower right', frameon=False, fontsize=60)  # Adjust the fontsize as needed
    plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.gca().set_yticklabels([])  # removes y-axis tick labels
    plt.ylim(0, 1)
    # if args.accuracy_type == "acc":
    #     plt.ylim(0.7, 1)
    # elif args.accuracy_type == "acc_far":
    #     plt.ylim(0.9, 1)
    # else: 
    #     plt.ylim(0, 1)
    plt.savefig(args.output_plot, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


# python plot_comm_acc.py \
#   --log_folder1 ./condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ./condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ./condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc \
#   --output_plot accuracy_plot_overall.pdf

# python plot_comm_acc.py \
#   --log_folder1 ./condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ./condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ./condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc_far \
#   --output_plot accuracy_plot_far.pdf

# python plot_comm_acc.py \
#   --log_folder1 ./condition_a/training_log_context --log_prefix1 log_rf \
#   --log_folder2 ./condition_b/training_log_context --log_prefix2 log_rf \
#   --log_folder3 ./condition_c/training_log_context --log_prefix3 log_rf \
#   --accuracy_type acc_close \
#   --output_plot accuracy_plot_close.pdf
