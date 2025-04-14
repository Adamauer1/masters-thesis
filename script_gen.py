import math
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
        # if placeholder == 'COLOR':
        #     color = get_random_gradient_color(value)
        #     content = content.replace(placeholder, str(color))
        #else:
        content = content.replace(placeholder, str(value))


    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

    print(f"File generated: {output_path}")



arousal_file_path = "Data/deam/arousal.csv"
arousal_df = pd.read_csv(arousal_file_path)
valence_file_path = "Data/deam/valence.csv"
valence_df = pd.read_csv(valence_file_path)

audio_features_df = pd.read_csv("Data/deam/2.csv", delimiter=";")
mask = (audio_features_df.index >= 30) & (audio_features_df.index <= 89)
audio_features_df = audio_features_df[mask]
#print(audio_features_df)
test = np.append(audio_features_df.columns.values, ["valence", "arousal"])

        

#audio_features_df = audio_features_df.to_dict()
#print(audio_features)
#test_af = audio_features[(audio_features[:, 0] >= 15) & (audio_features[:, 0] < 45)]
#print(af)
#audio_features = audio_features_df.values
#audio_features = audio_features_df.to_numpy()
#print(test_af)
test_v = valence_df.head(1).to_numpy()[0]
test_a = arousal_df.head(1).to_numpy()[0]
# test_v = valence_df.head(2).to_numpy()[1]
# test_a = arousal_df.head(2).to_numpy()[1]

test_a = test_a[~np.isnan(test_a)]
test_a = test_a[1::]
test_v = test_v[1::]
test_v = test_v[~np.isnan(test_v)]

af = dict.fromkeys(test[1::])
test_af = {}
row = audio_features_df.iloc[0]
#print(len(audio_features_df))
for i in range(len(audio_features_df)):
    row = audio_features_df.iloc[i]
    test_af[row["frameTime"]] = {
        "pcm_RMSenergy_sma_amean": row["pcm_RMSenergy_sma_amean"],
        "valence": test_v[i],
        "arousal": test_a[i]
    }

print(test_af)


#angles = np.degrees(np.arctan2(test_a[0::], test_v[0::])) % 360

#angle = math.degrees(math.atan2(test_a[0], test_v[0]))

def get_genre_color(genre):
    match genre:
        case "Blues": 
            return "(0,0,255)"
        case "Classical":
            return "(255,255,255)"
        case "Country":
            return "(0,255,0)"
        case "Hip-Hop":
            return "(255,0,0)"

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

def get_color(x, y):
    img = Image.open("tt.png")
    width, height = img.size

    pixel_x = int((x + 1) / 2 * width)
    pixel_y = int((1 - (y + 1) / 2) * height)
    return img.getpixel((pixel_x, pixel_y))

def get_colors(xs, ys):
    colors = []
    for index, x in enumerate(xs):
        colors.append(get_color(x,ys[index]))
    return colors


def generate_script(frames):
    for frame in frames:
        angle = math.degrees(math.atan2(frames[frame]["arousal"], frames[frame]["valence"]))
        frames[frame]["category"] = get_category(angle)
        frames[frame]["color_start"] = (0,0,255)
        frames[frame]["color_end"] = get_color(frames[frame]["valence"], frames[frame]["arousal"])

    print(frames)
    # frame time starts at 15 and goes to 44
    template_file = f"oldTemplates/{frames[25]["category"]}"
    new_file = "scripts/TestScript.jwfscript"
    placeholders = {
        "VALENCE": frames[25]["valence"] * 3,
        "AROUSAL": frames[25]["arousal"] * -2,
        "RED_END": frames[25]["color_end"][0],
        "GREEN_END": frames[25]["color_end"][1],
        "BLUE_END": frames[25]["color_end"][2],
        "RMSENERGY": frames[25]["pcm_RMSenergy_sma_amean"],
        "COLOR": frames[25]["color_start"]
    }
    
    write_script(template_file, new_file, placeholders)

generate_script(test_af)

