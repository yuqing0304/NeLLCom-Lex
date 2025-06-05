import matplotlib.pyplot as plt
import colorsys

# Define the HLS color values
hls_colors = [
    [88/360, 50/100, 50/100],  # Normalize H (0-1), L (0-1), S (0-1)
    [138/360, 50/100, 58/100],
    [150/360, 50/100, 79/100]
]

# Convert HLS to RGB for plotting or other use
rgb_colors = [colorsys.hls_to_rgb(h, l, s) for h, l, s in hls_colors]

# Save each color to a separate image file with dpi=300
for i, color in enumerate(rgb_colors, 1):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(f'{i}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid memory issues
