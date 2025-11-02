import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from collections import Counter
from plot_utils import extract_color_data, compute_cielab_distance, hls_to_rgb, rgb_to_cielab, compute_hls_distance, organize_data_model


def rename_files(input_files):
    """Rename the output files for each epoch"""
        
    files_to_rename = [('output_Initial Eval.txt', 'output_epoch0.txt')] + [
        (f'output_{i}.txt', f'output_epoch{i+1}.txt') for i in range(30)
    ]

    directory = os.path.dirname(input_files[0])  # Take the first file's directory

    # Rename each file in the list
    for old_file_name, new_file_name in files_to_rename:
        old_file_path = os.path.join(directory, old_file_name)
        
        if os.path.exists(old_file_path):
            new_file_path = os.path.join(directory, new_file_name)
            os.rename(old_file_path, new_file_path)


# ====================== Informativeness Computation ======================


def compute_informativeness_for_words(color_cielab_dict):
    informativeness = {}

    for word, cols in color_cielab_dict.items():
        # print(f"word, {word}")
        if len(cols) < 5:
            continue  # Skip if the word appears fewer than 5 times
        tmp = []
        if len(cols) < 100:
            for c in cols:
                for c2 in cols:
                    distance = compute_cielab_distance(c, c2)
                    if distance != 0.0:  # Only add non-zero distances
                        tmp.append(distance)
        else:
            tmp2 = []
            for i in range(30):  # Sample 100 pairs if there are more than 100 colors
                sampled = random.sample(cols, 100)
                for c in sampled:
                    for c2 in sampled:           
                        tmp2.append(compute_cielab_distance(c, c2))
                tmp2 = [i for i in tmp2 if i != 0.0]
                # print(f"tmp2, {tmp2}")
                tmp.append(sum(tmp2)/len(tmp2))
            # print(f"tmp, {tmp}")
        tmp = [i for i in tmp if i != 0.0]
        informativeness[word] = 100/(sum(tmp)/len(tmp))

    return informativeness





# def compute_informativeness_for_words_hls(color_hls_dict):
#     informativeness = {}

#     for word, cols in color_hls_dict.items():
#         # print(f"word, {word}")
#         if len(cols) < 3:
#             continue  # Skip if the word appears fewer than 5 times
#         tmp = []
#         if len(cols) < 100:
#             for c in cols:
#                 for c2 in cols:
#                     distance = compute_hls_distance(c, c2)
#                     if distance != 0.0:  # Only add non-zero distances
#                         tmp.append(distance)
#         else:
#             tmp2 = []
#             for i in range(30):  # Sample 100 pairs if there are more than 100 colors
#                 sampled = random.sample(cols, 100)
#                 for c in sampled:
#                     for c2 in sampled:           
#                         tmp2.append(compute_hls_distance(c, c2))
#                 tmp2 = [i for i in tmp2 if i != 0.0]
#                 # print(f"tmp2, {tmp2}")
#                 tmp.append(sum(tmp2)/len(tmp2))
#             # print(f"tmp, {tmp}")
#         tmp = [i for i in tmp if i != 0.0]
#         informativeness[word] = 100/(sum(tmp)/len(tmp))

#     return informativeness




def compute_lexical_system_informativeness(informativeness_df, words_used):
    """
    Computes lexical system-level informativeness as the sum of word informativeness (Iw) 
    for the words uttered to solve N interactions, divided by the number of interactions 
    (excluding those where the informativeness was not available).

    Args:
        informativeness_df (pd.DataFrame): DataFrame with 'color_name' and 'informativeness' columns.
        words_used (list): List of words used in interactions (one per interaction).

    Returns:
        float: Lexical system informativeness.
    """
    # Create a dictionary mapping color names to informativeness values
    informativeness_dict = dict(zip(informativeness_df["color_name"], informativeness_df["informativeness"]))

    # Retrieve informativeness for each word in words_used and skip if informativeness is 0 or word not found
    used_informativeness = [
        informativeness_dict[word] 
        for word in words_used 
        if word in informativeness_dict and informativeness_dict[word] > 0
    ]

    # Compute lexical informativeness as mean of non-zero informativeness values
    lexical_system_informativeness = sum(used_informativeness) / len(used_informativeness)

    return lexical_system_informativeness



# ====================== Informativeness Pipeline ======================
def process_file(file_path, rl_or_sl, label_or_agent):
    """Extracts color data, computes word informativeness, and saves separate CSVs for different conditions."""
    color_data_by_condition, words_used_by_condition, color_data, words_used = extract_color_data(file_path, rl_or_sl,use_label=label_or_agent)
    data_for_model = organize_data_model(file_path, rl_or_sl,use_label=label_or_agent)

    # Turn data_for_model into a DataFrame
    data_for_model_df = pd.DataFrame(data_for_model)
    
    # Accumulate HLS values per color name
    color_cielab_dict = {}
    # color_hls_dict = {}
    for cielab, color in color_data:
        # rgb = hls_to_rgb(*hls)
        # cielab = rgb_to_cielab(*rgb)
        color_cielab_dict.setdefault(color, []).append(cielab)
        # color_hls_dict.setdefault(color, []).append(hls)

    # Compute word informativeness
    # informativeness = compute_informativeness_for_words_hls(color_hls_dict)
    informativeness = compute_informativeness_for_words(color_cielab_dict)
    informativeness_df = pd.DataFrame(informativeness.items(), columns=["color_name", "informativeness"]).sort_values(by="informativeness", ascending=False)


    # ===== NEW: Merge informativeness into data_for_model_df =====
    # Create a dictionary for fast lookup
    informativeness_dict = dict(zip(informativeness_df["color_name"], informativeness_df["informativeness"]))

    # Map the 'name' column to informativeness
    data_for_model_df["informativeness"] = data_for_model_df["name"].map(informativeness_dict)

    # Save the data_for_model_df to CSV
    output_path = file_path.replace(".txt", "_model_data.csv")
    data_for_model_df.to_csv(output_path, index=False)


    informativeness_df.to_csv(file_path.replace(".txt", "_informativeness_overall.csv"), index=False)

    # informativeness_df = pd.read_csv("../dynamics/language_use/informativeness/dic_informativeness.csv").sort_values(by="informativeness", ascending=False)

    lexical_system_informativeness = compute_lexical_system_informativeness(informativeness_df, words_used)
    print(f"Overall Lexical System Informativeness for {file_path}: {lexical_system_informativeness}")


# ####### ====================== Informativeness by Condition begin : this is not correct ======================

    # for condition, color_data in color_data_by_condition.items():
    #     color_cielab_dict = {}

    #     for cielab, color in color_data:
    #         color_cielab_dict.setdefault(color, []).append(cielab)

    #     # Compute word informativeness
    #     informativeness = compute_informativeness_for_words(color_cielab_dict)
    #     informativeness_df = pd.DataFrame(informativeness.items(), columns=["color_name", "informativeness"]).sort_values(by="informativeness", ascending=False)

    #     # Save separate CSV per condition
    #     output_csv = file_path.replace(".txt", f"_informativeness_{condition}.csv")
    #     informativeness_df.to_csv(output_csv, index=False)
# ####### ====================== Informativeness by Condition begin : this is not correct ======================


    # ======= word used begin =======
    overall_words = []  # Collect all words for overall count
    condition_summary = []  # List to hold summary data for each condition

    for condition, color_data in color_data_by_condition.items():
        words = []
        for cielab, color in color_data:
            words.append(color)  # <-- change here
            # print(color, "\n")

        overall_words.extend(words)

        word_counts = Counter(words)
        word_used = pd.DataFrame(word_counts.items(), columns=["word", "count"])

        output_csv = file_path.replace(".txt", f"_word_used_{condition}.csv")
        word_used.to_csv(output_csv, index=False)

        # Track the number of words for this condition
        condition_summary.append({
            "condition": condition,
            "total_words": len(words),
            "unique_words": len(word_counts)
        })

    # ======= Add "overall" entry =======
    total_words_overall = len(overall_words)
    unique_words_overall = len(set(overall_words))

    condition_summary.append({
        "condition": "overall",
        "total_words": total_words_overall,
        "unique_words": unique_words_overall
    })

    # ======= Save overall word counts =======
    overall_counts = Counter(overall_words)
    overall_df = pd.DataFrame(overall_counts.items(), columns=["word", "count"])
    overall_output_csv = file_path.replace(".txt", f"_word_used_overall.csv")
    overall_df.to_csv(overall_output_csv, index=False)

    # ======= Summary of total words per condition =======
    summary_df = pd.DataFrame(condition_summary)
    summary_output_csv = file_path.replace(".txt", "_summary.csv")
    summary_df.to_csv(summary_output_csv, index=False)

    print(f"Summary CSV saved to {summary_output_csv}")
    # ======= word used end =======


    # Compute system-level informativenesss
    lexical_system_informativeness_by_condition = {}
    
    for condition, color_data in color_data_by_condition.items():
        lexical_system_informativeness = compute_lexical_system_informativeness(informativeness_df, words_used_by_condition[condition])
        lexical_system_informativeness_by_condition[condition] = lexical_system_informativeness

        print(f"Lexical System Informativeness for {condition}: {lexical_system_informativeness}")

    return lexical_system_informativeness_by_condition


# ====================== Batch Processing ======================
def process_multiple_files(file_paths, output_path, rl, label_or_agent):
    """Concatenates multiple files into one and processes them."""
    with open(output_path, 'w') as output_file:
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                output_file.write(f.read())

    print(f"Processing concatenated file: {output_path}")
    return process_file(output_path, rl, label_or_agent)
 



#=========================================correlate word informativeness============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlate_word_informativeness(computed_informativeness_file, reference_file, method):
    # Read the reference informativeness file
    reference_df = pd.read_csv(reference_file)

    # Read the computed informativeness file
    computed_df = pd.read_csv(computed_informativeness_file)

    # Merge both dataframes on color_name, keeping only rows with matching color names
    merged_df = computed_df.merge(reference_df, on="color_name", suffixes=("_computed", "_reference"))

    # Remove rows where either informativeness value is zero
    merged_df = merged_df[(merged_df["informativeness_computed"] > 0) & (merged_df["informativeness_reference"] > 0)]

    # Compute correlation
    # correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"])
    if method == "spearman":
        correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"], method="spearman")
    elif method == "pearson":
        correlation = merged_df["informativeness_computed"].corr(merged_df["informativeness_reference"], method="pearson")
    print(f"{method} correlation: {correlation:.4f}")

    # Scatterplot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=merged_df, x="informativeness_computed", y="informativeness_reference", alpha=0.7)

    # Add labels to the scatterplot
    for i in range(merged_df.shape[0]):
        plt.text(merged_df["informativeness_computed"].iloc[i],
                 merged_df["informativeness_reference"].iloc[i],
                 merged_df["color_name"].iloc[i],
                 fontsize=9, ha='right', va='bottom')

    # Set the same range for both axes
    min_val = min(merged_df["informativeness_computed"].min(), merged_df["informativeness_reference"].min())
    max_val = max(merged_df["informativeness_computed"].max(), merged_df["informativeness_reference"].max())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Computed Informativeness")
    plt.ylabel("Reference Informativeness")
    plt.title(f"Scatterplot of Informativeness (Correlation: {correlation:.2f})")
    plt.grid(True)
    output_plot_file = f'{computed_informativeness_file}.png'
    plt.savefig(output_plot_file)
    plt.close()


# ====================== Argument Parsing ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process color naming data and compute informativeness.")

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="List of input file paths (space-separated)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the concatenated output file."
    )
    parser.add_argument(
        "--use_label",
        action="store_true",
        help="Flag to use label (match.group(2)) instead of agent's word (match.group(1))."
    )

    parser.add_argument(
        "--rl_or_sl",
        action="store_true",
        help="Flag for rl or sl)."
    )

    parser.add_argument(
        "--correlation_method",
        type=str,
        choices=["pearson", "spearman"],
        default="pearson",
        help="Method for computing correlation (default: pearson)."
    )

    return parser.parse_args()


# ====================== Main Execution ======================
if __name__ == "__main__":
    args = parse_args()
    if args.rl_or_sl:
        rename_files(args.input_files)
    # Process the files with the specified arguments
    process_multiple_files(args.input_files, args.output_file, args.rl_or_sl, args.use_label)

    # reference_file = "../dynamics/language_use/informativeness/dic_informativeness.csv"
    # reference_file = "../condition1_human/dump_context/msg_spk_seed111/epochspk30_informativeness_overall.csv"

    correlation_method ="pearson"

    computed_informativeness_file = args.output_file.replace(".txt", "_informativeness_overall.csv")
    # if any("condition1_human" in path for path in args.input_files):
    #     print("right")
    #     correlate_word_informativeness(computed_informativeness_file, reference_file, correlation_method)



