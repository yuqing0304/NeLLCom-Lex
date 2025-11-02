import os
import pickle
import numpy as np
import csv
import argparse
from plot_utils import *


def extract_prototype(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return [(color_name, tuple(lab)) for color_name, lab in data.items()]


def differences_prototypes(data, data_human):
    data_dict = dict(data)
    human_dict = dict(data_human)
    common_names = set(data_dict.keys()) & set(human_dict.keys())
    differences = [
        compute_cielab_distance(data_dict[name], human_dict[name])
        for name in common_names
    ]
    return np.mean(differences)


def differences_prototypes_kept(data, data_human, last_epoch):
    kept_colors = [color_name for color_name, _ in last_epoch]
    results = []
    for color_name, cielab in data:
        for c_name, t_lab in data_human:
            if c_name == color_name and c_name in kept_colors:
                results.append(compute_cielab_distance(cielab, t_lab))
    return np.mean(results)


def analyze_differences(prototype_file, prototype_file_human, words_kept=False, last_epoch_num=None):
    prototype = extract_prototype(prototype_file)

    # pick last_epoch file depending on prototype_file type
    if 'rf' in prototype_file:
        if last_epoch_num is None:
            last_epoch_num = 30
        last_epoch_file = os.path.join(
            os.path.dirname(prototype_file),
            f'prototypes_rf_human_epoch{last_epoch_num}.pkl'
        )
    else:
        if last_epoch_num is None:
            last_epoch_num = 29
        last_epoch_file = os.path.join(
            os.path.dirname(prototype_file),
            f'prototypes_spk_human_epoch{last_epoch_num-1}.pkl'
        )

    last_epoch = extract_prototype(last_epoch_file)
    prototype_human = extract_prototype(prototype_file_human)

    if not words_kept:
        diff = differences_prototypes(prototype, prototype_human)
    else:
        diff = differences_prototypes_kept(prototype, prototype_human, last_epoch)

    print(diff)
    return diff



def compute_epoch_drift_across_seeds(epoch, seeds, label_prototype_path, base_dirs, output_file, last_epoch_num=None):
    results = []

    for condition, base_template in base_dirs.items():
        drifts = []
        drifts_kept = []

        for seed in seeds:
            base_path = base_template.format(seed=seed)
            prototype_file = os.path.join(base_path, f'prototypes_rf_human_epoch{epoch}.pkl')

            if not os.path.exists(prototype_file):
                print(f"Warning: File {prototype_file} not found.")
                continue

            drift = analyze_differences(prototype_file, label_prototype_path, last_epoch_num=last_epoch_num)
            drift_kept = analyze_differences(prototype_file, label_prototype_path, words_kept=True, last_epoch_num=last_epoch_num)

            drifts.append(drift)
            drifts_kept.append(drift_kept)

        n = len(drifts)
        mean_drift = np.mean(drifts)
        std_drift = np.std(drifts)
        se_drift = std_drift / np.sqrt(n)

        mean_drift_kept = np.mean(drifts_kept)
        std_drift_kept = np.std(drifts_kept)
        se_drift_kept = std_drift_kept / np.sqrt(n)

        results.append({
            "condition": condition,
            "epoch": epoch,
            "mean_drift": f"{mean_drift:.2f}",
            "std_drift": f"{std_drift:.2f}",
            "se_drift": f"{se_drift:.2f}",
            "mean_drift_kept": f"{mean_drift_kept:.2f}",
            "std_drift_kept": f"{std_drift_kept:.2f}",
            "se_drift_kept": f"{se_drift_kept:.2f}"
        })

    with open(output_file, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute prototype drift across seeds for multiple conditions.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number (e.g., 30)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seeds (e.g., 111 123 222)")
    parser.add_argument("--label_prototype", type=str, default="./prototypes.pkl", help="Path to human label prototype .pkl file")
    parser.add_argument("--base_dirs", nargs="+", required=True,
                        help="List of condition=path_template pairs, e.g., dump=../.../dump/msg_rf_seed{seed}/")
    parser.add_argument("--output", type=str, default="drift_summary.csv", help="Output CSV file")
    parser.add_argument("--last_epoch_num", type=int, default=None, help="Epoch number for last_epoch comparison (default: 30 for rf, 29 for spk)")

    args = parser.parse_args()

    # Parse base_dirs into a dict
    base_dirs = {}
    for entry in args.base_dirs:
        if "=" not in entry:
            raise ValueError(f"Invalid base_dir format: {entry}. Use condition=path_template")
        condition, path_template = entry.split("=", 1)
        base_dirs[condition] = path_template

    compute_epoch_drift_across_seeds(
        args.epoch,
        args.seeds,
        args.label_prototype,
        base_dirs,
        args.output,
        last_epoch_num=args.last_epoch_num
    )




