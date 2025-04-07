import math
import os
from PIL import Image

import numpy as np
import pandas as pd

def generate_script(template_path, output_path, placeholders=None):
    if placeholders is None:
        print("Placeholders is empty!")
        return

    with open(template_path, 'r', encoding='utf-8') as template_file:
        content = template_file.read()


    for placeholder, value in placeholders.items():
        content = content.replace(placeholder, str(value))


    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

    print(f"File generated: {output_path}")



# def load_and_label_points_np(file_path):
#     data = np.genfromtxt(file_path, delimiter=',', names=True)
#     sections = [get_section_label(x, y) for x, y in zip(data['x'], data['y'])]
#     labeled_data = np.core.records.fromarrays(
#         [data['x'], data['y'], sections],
#         names='x, y, section'
#     )
#     return labeled_data

arousal_file_path = "Data/deam/arousal.csv"
arousal_df = pd.read_csv(arousal_file_path)
valence_file_path = "Data/deam/valence.csv"
valence_df = pd.read_csv(valence_file_path)

# Display first few rows to confirm structure
#print(arousal_df.head())

test_v = valence_df.head(1).to_numpy()[0]
test_a = arousal_df.head(1).to_numpy()[0]
# test_v = valence_df.head(2).to_numpy()[1]
# test_a = arousal_df.head(2).to_numpy()[1]
test_a = test_a[:-1]
print(len(test_v))
print(len(test_a))
# test = np.column_stack((test_v, test_a))
#print(test)
angles = np.degrees(np.arctan2(test_v[1::], test_a[1::])) % 360

angle = math.degrees(math.atan2(test_v[1], test_a[1]))

cat = ""
#angle = angles[0]
if angle < 0:
    angle += 360

    if 0 <= angle < 45:
        cat = "JulianTest.jsfscript" #2
    elif 45 <= angle < 90:
        cat = "e-disc-gen.jwfscript" #1
    elif 90 <= angle < 135:
        cat = "synth-gen.jwfscript" #8
    elif 135 <= angle < 180:
        cat = "synth-gen.jwfscript" #7
    elif 180 <= angle < 225:
        cat = "linear-only-gen.jwfscript" #6
    elif 225 <= angle < 270:
        cat = "MandTest.jwfscript" #5
    elif 270 <= angle < 315:
        cat = "JulianTest.jwfscript" #4
    else:  # 315 <= angle < 360
        cat = "galaxies-gen.jwfscript" #3

print(test_v[2],test_a[2])
print(angle)
print(angles)
print(cat)

def normalized_to_pixel_coords(norm_x, norm_y, img_width, img_height):
    # Convert normalized coords (-1 to 1 range) to pixel coordinates
    px = int((norm_x + 1) / 2 * img_width)
    py = int((1 - (norm_y + 1) / 2) * img_height)  # Invert y-axis
    return px, py

# Load image
img = Image.open("testImage.png")
width, height = img.size

# Normalized input coordinates
norm_x, norm_y = test_v[1], test_a[1]

# Convert to pixel coordinates
px, py = normalized_to_pixel_coords(norm_x, norm_y, width, height)
print(f"Pixel coordinates: ({px}, {py})")

# Get RGB at that location
rgb = img.getpixel((px, py))
print(f"RGB value at ({px}, {py}): {rgb}")

template_file = f"scriptTemplates/{cat}"
new_file = "scripts/TestScript.jwfscript"
placeholders = {
    "VALENCE": test_v[1]*3,
    "AROUSAL": test_a[1]*-2,
    "RED": rgb[0],
    "GREEN": rgb[1],
    "BLUE": rgb[2],
    "RMSENERGY": 1.488783,
}



generate_script(template_file, new_file, placeholders)

