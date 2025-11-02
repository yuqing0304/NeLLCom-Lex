import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import os
import colorspacious as cs
import math


def cielab_to_rgb(lab):
    """Convert CIELAB to sRGB (0-1 range)."""
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb, 0, 1)
    return rgb


def visualize_cielab_colors(ax, cielab_colors, informativeness, word):
    l_values, a_values, b_values = zip(*cielab_colors)
    cielab_colors_array = np.array(cielab_colors)
    rgb_colors = cielab_to_rgb(cielab_colors_array)

    ax.scatter(a_values, b_values, l_values,
               facecolors=rgb_colors, s=50, marker='o')

    ax.set_xlim((-128, 127))
    ax.set_ylim((-128, 127))
    ax.set_zlim((0, 100))

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('L')
    ax.set_title(f"{word}, I={round(informativeness[word], 2)}")


def scatter_plot(ax, word, word_CIELABs, informativeness):
    cielabs = word_CIELABs[word]
    visualize_cielab_colors(ax, cielabs, informativeness, word)


def main(args):
    # Default seed list
    default_seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999,
                     123, 234, 345, 456, 567, 678, 789, 891, 912]
    seeds = args.seeds if args.seeds else default_seeds

    # Ensure output folders exist
    os.makedirs(args.out_dir, exist_ok=True)
    denotation_dir = os.path.join(args.out_dir, "denotation")
    os.makedirs(denotation_dir, exist_ok=True)

    for seed in seeds:
        path = os.path.join(args.base_path, f"msg_rf_seed{seed}", "epoch30_model_data.csv")
        # print(f"🔍 Processing seed {seed} from {path}")
        if not os.path.exists(path):
            print(f"⚠️ Skipping seed {seed}: no CSV found")
            continue

        # Load data
        data = pd.read_csv(path)
        word_CIELABs = {}
        informativeness = {}

        for _, row in data.iterrows():
            if row['name'] not in word_CIELABs:
                word_CIELABs[row['name']] = [ast.literal_eval(row['tar_cielab'])]
                informativeness[row['name']] = row['informativeness']
            else:
                word_CIELABs[row['name']].append(ast.literal_eval(row['tar_cielab']))

        words = list(word_CIELABs.keys())
        n = len(words)
        cols = 4
        rows = math.ceil(n / cols)

        # --- Big plot with all words
        fig = plt.figure(figsize=(cols*5, rows*5))
        for i, word in enumerate(words):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            scatter_plot(ax, word, word_CIELABs, informativeness)

        plt.tight_layout()
        out_big = os.path.join(args.out_dir, f"all_colors_seed{seed}.png")
        plt.savefig(out_big, dpi=300)
        plt.close(fig)
        print(f"✅ Saved big plot for seed {seed} → {out_big}")

        # --- Individual plots
        for word in words:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter_plot(ax, word, word_CIELABs, informativeness)
            plt.tight_layout()
            out_path = os.path.join(denotation_dir, f"{word}_seed{seed}.png")
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

        print(f"✅ Saved individual plots for seed {seed} in {denotation_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize color denotations in CIELAB space")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="List of seeds to process (default = predefined seed list)")
    parser.add_argument("--base_path", type=str, default="./dump_context",
                        help="Base directory containing msg_rf_seed*/epoch10_model_data.csv")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory to save plots (denotation folder will be created inside)")
    args = parser.parse_args()
    main(args)
