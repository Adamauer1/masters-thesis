import math
import os

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
test_a = test_a[:-1]
print(len(test_v))
print(len(test_a))
test = np.column_stack((test_v, test_a))
print(test[1])

angle = math.degrees(math.atan2(test[2][1], test[1][0]))

cat = ""

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

print(angle)
print(cat)

template_file = f"scriptTemplates/{cat}"
new_file = "scripts/TestScript.jwfscript"
placeholders = {
    "PLACEHOLDER_1": 1,
    "PLACEHOLDER_2": 2
}

#generate_script(template_file, new_file, placeholders)

