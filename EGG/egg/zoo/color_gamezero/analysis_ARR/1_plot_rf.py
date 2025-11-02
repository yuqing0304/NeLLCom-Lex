import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd

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

def calculate_95ci(accs, unique_epochs):
    """
    Calculates the 95% confidence interval for each epoch's accuracy data.

    Parameters:
        accs (numpy array): Accuracy values across all epochs.
        unique_epochs (numpy array): Unique epochs to calculate the CI.

    Returns:
        tuple: average accuracy and confidence interval (lower, upper).
    """
    avg_accs = []
    lower_ci = []
    upper_ci = []

    for epoch in unique_epochs:
        epoch_indices = accs[0] == epoch  # Assumes all accuracy lists are aligned by epoch
        n = np.sum(epoch_indices)  # Number of data points (seeds)
        avg_acc = np.mean(accs[1][epoch_indices])
        std_acc = np.std(accs[1][epoch_indices])

        # Calculate the 95% confidence interval
        ci = 1.96 * (std_acc / np.sqrt(n))

        avg_accs.append(avg_acc)
        lower_ci.append(avg_acc - ci)
        upper_ci.append(avg_acc + ci)

    return np.array(avg_accs), np.array(lower_ci), np.array(upper_ci)

def read_and_plot_accuracy(log_folder, log_prefix, label, accuracy_type):
    """
    Reads JSON log files with the specified prefix from a folder, extracts the specified accuracy metrics,
    calculates the average and 95% confidence intervals over all seeds, and plots it as a function of epoch.
    """
    # Collect all log files matching the prefix
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

        # Read the file and extract epoch and all accuracy types
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry.get("mode") == "test":  # Filter for test mode
                        epoch = log_entry["epoch"]
                        valid_acc = log_entry["acc"]
                        far_acc = log_entry["acc_far"]
                        close_acc = log_entry["acc_close"]
                        split_acc = log_entry["acc_split"]

                        epochs.append(epoch)
                        valid_accs.append(valid_acc)
                        far_accs.append(far_acc)
                        close_accs.append(close_acc)
                        split_accs.append(split_acc)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in file: {log_file}")

        if epochs and valid_accs:
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

    if not all_epochs or not all_valid_accs:
        print(f"No valid data found in {log_folder} with prefix {log_prefix}.")
        return

    all_epochs = np.concatenate(all_epochs)
    all_valid_accs = np.concatenate(all_valid_accs)
    all_far_accs = np.concatenate(all_far_accs)
    all_close_accs = np.concatenate(all_close_accs)
    all_split_accs = np.concatenate(all_split_accs)

    unique_epochs = np.unique(all_epochs)

    # Calculate 95% CI for all accuracy types
    accs_data = [(all_epochs, all_valid_accs), (all_epochs, all_far_accs), (all_epochs, all_close_accs), (all_epochs, all_split_accs)]
    if accuracy_type == 'valid':
        avg_accs, lower_ci, upper_ci = calculate_95ci(accs_data[0], unique_epochs)
        label_text = f'{label} - Valid Acc'
    elif accuracy_type == 'far':
        avg_accs, lower_ci, upper_ci = calculate_95ci(accs_data[1], unique_epochs)
        label_text = f'{label} - Far Acc'
    elif accuracy_type == 'close':
        avg_accs, lower_ci, upper_ci = calculate_95ci(accs_data[2], unique_epochs)
        label_text = f'{label} - Close Acc'
    elif accuracy_type == 'split':
        avg_accs, lower_ci, upper_ci = calculate_95ci(accs_data[3], unique_epochs)
        label_text = f'{label} - Split Acc'
    else:
        raise ValueError(f"Unknown accuracy type: {accuracy_type}")

    # Plot the accuracy with shaded region for 95% CI
    # colors = {
    #     "SL w/o context (zeroed) + RL w/o context": "#95a3c5",
    #     "SL w/ context + RL w/ context": "#d4d4d4",
    #     "SL w/o context (zeroed) + RL w/ context": "#f3b8ad"
    # }

    colors = {
        "SL w/o context (zeroed) + RL w/o context": "#b0b0b0",  
        "SL w/ context + RL w/ context": "#3b4c6b",            
        "SL w/o context (zeroed) + RL w/ context": "#f2a59b"     
    }

    label_renames = {
        "SL w/o context (zeroed) + RL w/o context": r"$\mathrm{SL{-}RL{-}}$",
        "SL w/ context + RL w/ context":            r"$\mathrm{SL{+}RL{+}}$",
        "SL w/o context (zeroed) + RL w/ context":  r"$\mathrm{SL{-}RL{+}}$",
    }


    renamed_label = label_renames.get(label, label)
    plt.plot(unique_epochs, avg_accs, label=renamed_label, color=colors[label], linewidth=5, markeredgewidth=0)
    plt.fill_between(unique_epochs, lower_ci, upper_ci, color=colors[label], alpha=0.3)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot validation accuracy from training logs.")
    parser.add_argument('--log_folder1', type=str, required=True, help="First folder containing log files")
    parser.add_argument('--log_prefix1', type=str, required=True, help="Prefix of the first log files")
    parser.add_argument('--log_folder2', type=str, required=True, help="Second folder containing log files")
    parser.add_argument('--log_prefix2', type=str, required=True, help="Prefix of the second log files")
    parser.add_argument('--log_folder3', type=str, required=True, help="Third folder containing log files")
    parser.add_argument('--log_prefix3', type=str, required=True, help="Prefix of the third log files")
    parser.add_argument('--accuracy_type', type=str, choices=['valid', 'far', 'close', 'split'], required=True, help="Accuracy type to plot (valid, far, close, split)")
    parser.add_argument('--output_plot', type=str, required=True, help="Output plot file name")

    # Parse the arguments
    args = parser.parse_args()

    # Plot for each accuracy type (valid, far, close, split) for each condition
    read_and_plot_accuracy(args.log_folder1, args.log_prefix1, "SL w/o context (zeroed) + RL w/o context", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder2, args.log_prefix2, "SL w/ context + RL w/ context", args.accuracy_type)
    read_and_plot_accuracy(args.log_folder3, args.log_prefix3, "SL w/o context (zeroed) + RL w/ context", args.accuracy_type)

    # plt.xlabel("Epoch", labelpad=10)
    plt.ylabel("Comm Acc", labelpad=10)

    if "human" in args.log_folder1:
        plot_type = "human"
    elif "generated" in args.log_folder1:
        plot_type = "generated"
    else:
        plot_type = "dataset"

    plt.xticks(np.arange(0, 31, 10))
    plt.ylim(0, 1)
    plt.legend(loc='lower right', frameon=False, fontsize=60)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().set_yticklabels([])  # removes y-axis tick labels
    plt.tight_layout()

    # Save the plot to a file
    parent_output_dir = os.path.abspath(os.path.join(args.log_folder1, os.pardir))
    output_path = os.path.join(parent_output_dir, args.output_plot)

    # Save the plot to the constructed path
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
