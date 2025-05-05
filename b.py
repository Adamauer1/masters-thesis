# import moviepy
# 
# # Load the audio file
# clip = moviepy.AudioFileClip("audio/test.mp3")
# 
# # Cut from 0s to 30s
# short_clip = clip.subclipped(30, 60)
# 
# # Write to file
# short_clip.write_audiofile("output_30sec.mp3")

import matplotlib.pyplot as plt
import numpy as np

def interpolate_color_255(a, b, t):
    """Linearly interpolate between two RGB colors in 0-255 range."""
    return tuple(int((1 - t) * a[i] + t * b[i]) for i in range(3))

def biased_gradient_255(a, b, steps=100, bias_strength=3.0):
    """
    Generate a biased gradient from color a to b using 0-255 RGB values.
    bias_strength > 1 favors color a.
    """
    gradient = []
    for i in range(steps):
        t = (i / (steps - 1)) ** bias_strength
        gradient.append(interpolate_color_255(a, b, t))
    return gradient

# Example RGB colors (0–255)
color_a = (51, 102, 204)   # A medium blue
color_b = (255, 51, 51)    # A strong red

# Generate gradient
gradient = biased_gradient_255(color_a, color_b, steps=100, bias_strength=3.0)

# Optional: visualize it using matplotlib
def show_gradient(gradient):
    # Convert to 0–1 for display
    normalized = [(r / 255, g / 255, b / 255) for r, g, b in gradient]
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([normalized], extent=[0, 10, 0, 1])
    ax.set_axis_off()
    plt.show()

show_gradient(gradient)

# If needed, you can print or use the `gradient` list directly