import numpy 

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)


import numpy as np
import random
import colorsys
import colorspacious as cs
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab

# Function to convert HLS to LAB
def hls_to_lab(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    rgb = np.array([r, g, b]).reshape(1, 1, 3)
    lab = rgb2lab(rgb)
    return lab.reshape(-1)

# Function to compute CIEDE2000 color difference
def ciede_distance(c1, c2):
    # Convert RGB to Lab
    c1_lab = convert_color(sRGBColor(*c1, is_upscaled=False), LabColor)
    c2_lab = convert_color(sRGBColor(*c2, is_upscaled=False), LabColor)
    # Compute CIEDE2000 distance
    return delta_e_cie2000(c1_lab, c2_lab)


def cielab_to_rgb(lab):
    """Convert CIELAB to sRGB (0-1 range)."""
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb, 0, 1)  # Ensure values are within valid RGB range
    return rgb

# Function to generate a random base color in HLS
def sample_base_color(category):
    if category == "red":
        return np.array([random.choice([random.randint(0, 60), random.randint(300, 360)]), 50, random.randint(50, 100)])
    elif category == "green":
        return np.array([random.randint(60, 180), 50, random.randint(50, 100)])
    elif category == "blue":
        return np.array([random.randint(180, 300), 50, random.randint(50, 100)])


def generate_distractors(base_color_lab, condition, theta, epsilon):
    distractors = []

    if condition == "close":
        while len(distractors) < 2:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            if epsilon <= distance <= theta:
                distractors.append(new_color_lab)

    elif condition == "far":
        while len(distractors) < 2:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            if distance > theta:
                distractors.append(new_color_lab)

    elif condition == "split":
        close_color = None
        far_color = None
        while close_color is None or far_color is None:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            if epsilon <= distance <= theta and close_color is None:
                close_color = new_color_lab
            elif distance > theta and far_color is None:
                far_color = new_color_lab
        distractors = [close_color, far_color]

    return distractors


def generate_trial(condition, theta, epsilon):
    """Generates a trial ensuring diverse and balanced base colors across R, G, B."""
    
    # Randomly select one of the three main color categories
    base_category = random.choice(["red", "green", "blue"])
    
    # Sample a base color from that category
    base_color_hls = sample_base_color(base_category)
    
    # Convert to LAB color space for perceptual distance calculation
    base_color_lab = hls_to_lab(*base_color_hls)

    colors = []
    
    if condition == "close":
        # Generate 2 additional colors close to the base color (within θ but above ε)
        while len(colors) < 2:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            
            if epsilon <= distance <= theta:  # Ensure perceptual constraint
                colors.append(new_color_lab)
    
    elif condition == "split":
        close_color = None
        far_color = None
        
        while close_color is None or far_color is None:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            if epsilon <= distance <= theta and close_color is None:
                close_color = new_color_lab
            elif distance > theta and far_color is None:
                far_color = new_color_lab
        colors = [close_color, far_color]
    
    elif condition == "far":
        while len(colors) < 2:
            new_color_hls = np.array([random.randint(0, 360), 50, random.randint(50, 100)])
            new_color_lab = hls_to_lab(*new_color_hls)
            distance = deltaE_ciede2000(base_color_lab, new_color_lab)
            if distance > theta:
                colors.append(new_color_lab)

    colors.insert(0, base_color_lab)
    return np.array(colors)



def generate_trials(num_trials, theta, epsilon):
    trials = {"close": [], "split": [], "far": []}
    color_names = ["none"]

    for _ in range(num_trials):
        base_category = random.choice(["red", "green", "blue"])
        base_color_hls = sample_base_color(base_category)
        base_color_lab = hls_to_lab(*base_color_hls)

        for condition in ["close", "split", "far"]:
            distractors = generate_distractors(base_color_lab, condition, theta, epsilon)
            if len(distractors) != 2:
                continue  # Skip if generation failed for some reason

            trial_colors = [base_color_lab] + distractors
            flattened_trial = np.array(trial_colors).reshape(-1, 3)

            # Validation
            base_lab = flattened_trial[0]
            d1_lab = flattened_trial[1]
            d2_lab = flattened_trial[2]
            d1_dist = deltaE_ciede2000(base_lab, d1_lab)
            d2_dist = deltaE_ciede2000(base_lab, d2_lab)

            # Skip if not satisfying constraints (shouldn't happen, but safety check)
            if condition == "close" and not (epsilon <= d1_dist < theta and epsilon <= d2_dist < theta):
                continue
            elif condition == "far" and not (d1_dist > theta and d2_dist > theta):
                continue
            elif condition == "split":
                close_ok = (epsilon <= d1_dist < theta or epsilon <= d2_dist < theta)
                far_ok = (d1_dist > theta or d2_dist > theta)
                if not (close_ok and far_ok):
                    continue

            shuffled = flattened_trial.tolist()
            random.shuffle(shuffled)

            target_color = tuple(int(x) for x in flattened_trial[0])
            shuffled_colors = [tuple(int(x) for x in color) for color in shuffled]
            position = shuffled_colors.index(target_color)
            color_name = random.choice(color_names)

            trials[condition].append((
                f"\"[{', '.join(map(str, map(int, flattened_trial[0])))}]\"",
                f"\"[{', '.join(map(str, map(int, flattened_trial[1])))}]\"",
                f"\"[{', '.join(map(str, map(int, flattened_trial[2])))}]\"",
                f"\"[{', '.join(map(str, map(int, shuffled[0])))}]\"",
                f"\"[{', '.join(map(str, map(int, shuffled[1])))}]\"",
                f"\"[{', '.join(map(str, map(int, shuffled[2])))}]\"",
                position,
                color_name
            ))

    return trials

# Parameters
theta = 20  # Threshold for condition
epsilon = 5  # Minimum perceptible difference
num_trials = 30000  # Number of trials per condition


# Generate trials
trials = generate_trials(num_trials, theta, epsilon)

# Function to output trials to separate text files
def output_trials_to_files(trials):
    for condition, condition_trials in trials.items():
        # Open a new file for each condition
        with open(f"sample_{condition}.txt", "w") as file:
            for trial in condition_trials:
                file.write(",".join(map(str, trial)) + "\n")

# Output trials to separate text files
output_trials_to_files(trials)



def hls_to_rgb(h, l, s):
    """Convert HLS to RGB."""
    # Normalize the HLS values
    h = h / 360.0  # Hue should be between 0 and 1
    l = l / 100.0  # Lightness should be between 0 and 1
    s = s / 100.0  # Saturation should be between 0 and 1

    # Convert using colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    # Return the RGB as a tuple in the range [0, 1] (matplotlib compatible)
    return (r, g, b)

def visualize_and_save_trials(trials, max_trials=100):
    for condition, condition_trials in trials.items():
        # Limit the number of trials to visualize
        condition_trials = condition_trials[:max_trials]

        fig, ax = plt.subplots(len(condition_trials), 1, figsize=(10, len(condition_trials) * 2))
        
        if len(condition_trials) == 1:
            ax = [ax]
        
        ax[0].set_title(f"{condition.capitalize()} Condition")
        
        for trial_idx, trial in enumerate(condition_trials):
            # Extract the colors from the trial (assumes HLS format)
            colors_cielab = [
                tuple(map(int, color.strip('"[]').split(", "))) for color in trial[:6]
            ]
            
            # Convert HLS to RGB
            colors_rgb = [cielab_to_rgb(lab) for lab in colors_cielab]
           
            # Create a 2D array for imshow
            colors_array = [colors_rgb]  # List of RGB colors
            
            # Plot the colors as patches
            ax[trial_idx].imshow(colors_array, aspect="auto")
            ax[trial_idx].axis("off")  # Hide the axes for a clean display
        
        # Save the image for the current condition
        plt.tight_layout()
        plt.savefig(f"trials_{condition}.png")
        plt.close()

# Visualize and save the first 100 trials for each condition
visualize_and_save_trials(trials, max_trials=200)

