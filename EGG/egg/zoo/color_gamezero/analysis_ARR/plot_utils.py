import ast 
import colorspacious as cs
import re
import colorsys
import numpy as np 
import pickle 
import csv
import ast
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
import numpy as np

if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

def organize_data_model(file_path, rl=False, use_label=False):
    results = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 2:
                continue  # Skip malformed lines

            try:
                cielab_values = ast.literal_eval(parts[0].strip())  # Convert cielab string to list
            except (SyntaxError, ValueError):
                continue  # Skip invalid lines
            color_part = parts[1].strip()
            if rl: 
                match = re.match(r"(\w+)\s*\((.*?)\)", color_part)
                condition = parts[5].strip()
                outcome_1 = parts[3].strip()
                outcome_2 = parts[4].strip()
                # print(f"outcome_1, {outcome_1}")
                # print(f"outcome_2, {outcome_2}")
                # Skip unsuccessful trials
                if outcome_1 != outcome_2:
                    continue                
            else:
                match = re.match(r"(\w+)\s*\(label=(.*?)\)", color_part)
                condition = parts[2].strip()
                outcome_1 = None 
                outcome_2 = None 

            agent_color = match.group(1)
            label_color = match.group(2)
            tar_cielab = cielab_values[0]
            # target = LabColor(lab_l=tar_cielab[0], lab_a=tar_cielab[1], lab_b=tar_cielab[2])
            dist1_cielab = cielab_values[1]
            dist2_cielab = cielab_values[2]

          # Create LabColor objects
            target = LabColor(*tar_cielab)
            distractor1 = LabColor(*dist1_cielab)
            distractor2 = LabColor(*dist2_cielab)

            # Calculate distances and round to nearest integer
            targetD1Diff = int(round(delta_e_cie2000(target, distractor1)))
            targetD2Diff = int(round(delta_e_cie2000(target, distractor2)))
            D1D2Diff = int(round(delta_e_cie2000(distractor1, distractor2)))


            extracted_name = label_color if use_label else agent_color

            # Hard and easy distractors
            hard_diff = min(targetD1Diff, targetD2Diff)
            easy_diff = max(targetD1Diff, targetD2Diff)

            results.append({
                "tar_cielab": tar_cielab,
                "dist1_cielab": dist1_cielab,
                "dist2_cielab": dist2_cielab,
                "name": extracted_name,
                "condition": condition,
                "outcome_1": outcome_1,
                "outcome_2": outcome_2,
                "targetD1Diff": targetD1Diff,
                "targetD2Diff": targetD2Diff,
                "D1D2Diff": D1D2Diff,
                "hard_diff": hard_diff,
                "easy_diff": easy_diff
            })

    return results




def extract_color_data_all_label(file_path):
    """
    Extract cielab values and color names (or labels) from a CSV-style file.

    Args:
        file_path (str): Path to the input file.
        rl (bool): Not used in this version, but kept for compatibility.
        use_label (bool): If True, extract label (currently unused since no label field exists in format).

    Returns:
        tuple: (color_data_by_condition, words_used_by_condition, all_color_data, all_words_used)
    """
    words_used_by_condition = {}
    color_data_by_condition = {}

    all_words_used = []
    all_color_data = []

    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for parts in reader:
            try:
                first_color = ast.literal_eval(parts[0])
            except Exception:
                continue  # Skip invalid lines

            color_name = parts[7].strip()
            condition = parts[8].strip().lower()
            extracted_name = color_name  # 'label' is not used in this structure
            # Initialize condition-specific lists
            if condition not in words_used_by_condition:
                words_used_by_condition[condition] = []
                color_data_by_condition[condition] = []

            words_used_by_condition[condition].append(extracted_name)
            color_data_by_condition[condition].append((first_color, extracted_name))

            all_words_used.append(extracted_name)
            all_color_data.append((first_color, extracted_name))

    return color_data_by_condition, words_used_by_condition, all_color_data, all_words_used



def extract_color_data(file_path, rl=False, use_label=False):
    """
    Extract cielab values and color names (or labels) from a file.
    
    Args:
        file_path (str): Path to the input file.
        use_label (bool): If True, extract label (match.group(2)). Otherwise, extract word (match.group(1)).

    Returns:
        tuple: (color_data, words_used)
            - color_data: List of (cielab_value, extracted_name) pairs.
            - words_used: List of extracted names used in successful trials.
    """
    words_used_by_condition = {}  # Track words used in successful trials per condition
    color_data_by_condition = {}  # Store extracted (cielab, color_name) pairs per condition

    all_words_used = []
    all_color_data = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split("->")
            if len(parts) < 2:
                continue  # Skip malformed lines

            try:
                cielab_values = ast.literal_eval(parts[0].strip())  # Convert cielab string to list
            except (SyntaxError, ValueError):
                continue  # Skip invalid lines
            color_part = parts[1].strip()
            if rl: 
                match = re.match(r"(\w+)\s*\((.*?)\)", color_part)
                condition = parts[5].strip()
                outcome_1 = parts[3].strip()
                outcome_2 = parts[4].strip()
                # print(f"outcome_1, {outcome_1}")
                # print(f"outcome_2, {outcome_2}")
                # Skip unsuccessful trials
                if outcome_1 != outcome_2:
                    continue                
            else:
                match = re.match(r"(\w+)\s*\(label=(.*?)\)", color_part)
                condition = parts[2].strip()

            agent_color = match.group(1)
            label_color = match.group(2)

            
            extracted_name = label_color if use_label else agent_color


            # Initialize condition-specific lists
            if condition not in words_used_by_condition:
                words_used_by_condition[condition] = []
                color_data_by_condition[condition] = []

            words_used_by_condition[condition].append(extracted_name)
            color_data_by_condition[condition].append((cielab_values[0], extracted_name))  # Use first cielab value

            all_words_used.append(extracted_name)
            all_color_data.append((cielab_values[0], extracted_name))

    return color_data_by_condition, words_used_by_condition, all_color_data, all_words_used



def rgb_to_cielab(r, g, b):
    """Convert RGB (0-255) to CIELAB."""
    return cs.cspace_convert((r, g, b), start="sRGB255", end="CIELab")



# Function to convert HLS to RGB
def hls_to_rgb(h, l, s):
    h = h / 360.0  # Scale hue to [0, 1]
    l = l / 100.0  # Scale lightness to [0, 1]
    s = s / 100.0  # Scale saturation to [0, 1]
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))



# Function to calculate CIELAB color distance
def compute_cielab_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))


def compute_hls_distance(hls1, hls2):
    return np.linalg.norm(np.array(hls1) - np.array(hls2))



def load_prototypes(pickle_path):
    """Load color prototypes from a pickle file."""
    with open(pickle_path, "rb") as f:
        prototypes = pickle.load(f)
    return prototypes


def cielab_to_rgb(lab):
    """Convert CIELAB to sRGB (0-1 range)."""
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb, 0, 1)  # Ensure values are within valid RGB range
    return rgb