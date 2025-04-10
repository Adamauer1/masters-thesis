import math
import os
from PIL import Image

import numpy as np
import pandas as pd

def write_script(template_path, output_path, placeholders=None):
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



arousal_file_path = "Data/deam/arousal.csv"
arousal_df = pd.read_csv(arousal_file_path)
valence_file_path = "Data/deam/valence.csv"
valence_df = pd.read_csv(valence_file_path)

audio_features_df = pd.read_csv("Data/deam/2.csv", delimiter=";")
#print(audio_features_df)
#audio_features = audio_features_df.values
audio_features = audio_features_df.to_numpy()
#print(audio_features)
test_af = audio_features[(audio_features[:, 0] >= 15) & (audio_features[:, 0] < 45)]
print(len(test_af))
test_v = valence_df.head(1).to_numpy()[0]
test_a = arousal_df.head(1).to_numpy()[0]
# test_v = valence_df.head(2).to_numpy()[1]
# test_a = arousal_df.head(2).to_numpy()[1]

test_a = test_a[~np.isnan(test_a)]
test_a = test_a[1::]
test_v = test_v[1::]
test_v = test_v[~np.isnan(test_v)]

# print(len(test_v))
# print(len(test_a))

# test = np.column_stack((test_v, test_a))
#print(test)

#angles = np.degrees(np.arctan2(test_a[0::], test_v[0::])) % 360

#angle = math.degrees(math.atan2(test_a[0], test_v[0]))

def get_category(angle):
    if angle < 0:
        angle += 360

    if 0 <= angle < 45:
        return "JulianTest.jsfscript"  # 2
    elif 45 <= angle < 90:
        return "e-disc-gen.jwfscript"  # 1
    elif 90 <= angle < 135:
        return "synth-gen.jwfscript"  # 8
    elif 135 <= angle < 180:
        return "synth-gen.jwfscript"  # 7
    elif 180 <= angle < 225:
        return "linear-only-gen.jwfscript"  # 6
    elif 225 <= angle < 270:
        return "MandTest.jwfscript"  # 5
    elif 270 <= angle < 315:
        return "JulianTest.jwfscript"  # 4
    else:  # 315 <= angle < 360
        return "galaxies-gen.jwfscript"  # 3

def get_categories(angles):
    categories = []
    for angle in angles:
        categories.append(get_category(angle))
    return categories

#cats = get_categories(angles)

np.set_printoptions(threshold=np.inf)
#print(test_v,test_a)
#print(angle)
#print(angles)
#print(cats)

# def normalized_to_pixel_coords(norm_x, norm_y, img_width, img_height):
#     # Convert normalized coords (-1 to 1 range) to pixel coordinates
#     px = int((norm_x + 1) / 2 * img_width)
#     py = int((1 - (norm_y + 1) / 2) * img_height)  # Invert y-axis
#     return px, py

# # Load image
# img = Image.open("testImage.png")
# width, height = img.size
#
# # Normalized input coordinates
# norm_x, norm_y = test_v[-1], test_a[-1]
#
# # Convert to pixel coordinates
# px, py = normalized_to_pixel_coords(norm_x, norm_y, width, height)
# print(f"Pixel coordinates: ({px}, {py})")

def get_color(x, y):
    img = Image.open("testImage.png")
    width, height = img.size

    pixel_x = int((x + 1) / 2 * width)
    pixel_y = int((1 - (y + 1) / 2) * height)
    return img.getpixel((pixel_x, pixel_y))

def get_colors(xs, ys):
    colors = []
    for index, x in enumerate(xs):
        colors.append(get_color(x,ys[index]))
    return colors
# Get RGB at that location
#rgb = img.getpixel((px, py))
#rgb = get_colors(test_v, test_a)
#print(rgb)

# template_file = f"scriptTemplates/{cats[-1]}"
# new_file = "scripts/TestScript.jwfscript"
# placeholders = {
#     "VALENCE": test_v[1]*3,
#     "AROUSAL": test_a[1]*-2,
#     "RED": rgb[0],
#     "GREEN": rgb[1],
#     "BLUE": rgb[2],
#     "RMSENERGY": 1.488783,
# }


def generate_script(valence_values, arousal_values):
    angles = np.degrees(np.arctan2(arousal_values, valence_values)) % 360
    categories = get_categories(angles)
    colors = get_colors(valence_values, arousal_values)

    print(angles)
    print(categories)
    print(colors)

    template_file = f"scriptTemplates/{categories[-1]}"
    new_file = "scripts/TestScript.jwfscript"
    placeholders = {
        "VALENCE": valence_values[0] * 3,
        "AROUSAL": arousal_values[0] * -2,
        "RED": colors[0][0],
        "GREEN": colors[0][1],
        "BLUE": colors[0][2],
        "RMSENERGY": 1.488783,
    }
    write_script(template_file, new_file, placeholders)

#generate_script(test_v, test_a)

#write_script(template_file, new_file, placeholders)
