import math
from PIL import Image
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.utils import column_or_1d
import matplotlib.image as mpimg

from features_extraction import extract_features, extract_bpm


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


def get_genre_color(genre):
    match genre:
        case "blues": 
            return [0,0,255]
        case "classical":
            return [255,255,255]
        case "country":
            return [0,255,0]
        case "hiphop":
            return [255,0,0]
        case "disco":
            return [0,255,255]
        case "jazz":
            return [128,128,128]
        case "metal":
            return [105,105,105]
        case "pop":
            return [255,192,203]
        case "reggae":
            return [255, 165, 0]
        case "rock":
            return [255,0,0]

def get_category(angle):
    if angle < 0:
        angle += 360

    if 0 <= angle < 45:
        return "JulianGen"  # 2
    elif 45 <= angle < 90:
        return "BrokatGen"  # 1
    elif 90 <= angle < 135:
        return "SynthGen"  # 8
    elif 135 <= angle < 180:
        return "SynthGen"  # 7
    elif 180 <= angle < 225:
        return "LinearOnlyGen"  # 6
    elif 225 <= angle < 270:
        return "MandGen"  # 5
    elif 270 <= angle < 315:
        return "JulianGen"  # 4
    else:  # 315 <= angle < 360
        return "GalaxiesGen"  # 3

def get_categories(angles):
    categories = []
    for angle in angles:
        categories.append(get_category(angle))
    return categories

def get_color(x, y):
    img = mpimg.imread('tt.png')
    img_height, img_width, _ = img.shape

# Calculate the center of the image
    center_x = img_width / 2
    center_y = img_height / 2
    #print(center_x)
    #print(center_y)

# Define the translated coordinate mappings
    translated_coords = [
        (center_x - 450, center_y - 450),
        (center_x + 450, center_y - 450),
        (center_x - 450, center_y + 450),
        (center_x + 450, center_y + 450)
    ]
    angle = math.atan2(x, y)
    #print(angle)
#angle2 = math.atan( 0.03808333333/0.08258666666)
#print(angle2) 
#color3 = get_color(math.cos(angle), math.sin(angle))
# 2. Example Original Data Points
    x_original = [math.cos(angle)]
    y_original = [math.sin(angle)]

    def translate_coordinate_centered(x_orig, y_orig):
        scale_x = translated_coords[1][0] - translated_coords[0][0]
        scale_y = translated_coords[2][1] - translated_coords[0][1]
        x_translated = center_x + x_orig * (scale_x / 2)
        y_translated = center_y - y_orig * (scale_y / 2)
        return int(round(x_translated)), int(round(y_translated))

    rgb_values_0_255 = []
    for x_orig, y_orig in zip(x_original, y_original):
        x_pixel, y_pixel = translate_coordinate_centered(x_orig, y_orig)

        if 0 <= y_pixel < img_height and 0 <= x_pixel < img_width:
            rgb = img[y_pixel, x_pixel]
            # Scale the RGB values from 0-1 to 0-255
            rgb_255 = (rgb * 255).astype(int)
            rgb_values_0_255.append(rgb_255)
            #print(f"Original: ({x_orig:.3f}, {y_orig:.3f}), Pixel: ({x_pixel}, {y_pixel}), RGB (0-255): {rgb_255}")
        else:
            rgb_values_0_255.append(None)
            #print(f"Original: ({x_orig:.3f}, {y_orig:.3f}), Pixel: ({x_pixel}, {y_pixel}), Out of bounds")
    return rgb_values_0_255[0]


def get_colors(xs, ys):
    colors = []
    for index, x in enumerate(xs):
        colors.append(get_color(x,ys[index]))
    return colors


def generate_script(song_data, output_script_name):
    video_frames = {}
    frame_index = 0
    reset_frame_counter = 0
    last_category = ""
    energy_mean = np.float64(0)
    audio_frames = song_data["audio_frames"]
    
    for audio_frame in audio_frames:
        angle = math.atan2(audio_frames[audio_frame]["arousal"], audio_frames[audio_frame]["valence"])
        audio_frames[audio_frame]["startTime"] = audio_frame
        audio_frames[audio_frame]["category"] = get_category(math.degrees(angle))
        audio_frames[audio_frame]["color_start"] = get_genre_color(song_data["genre"])
        audio_frames[audio_frame]["color_end"] = get_color(audio_frames[audio_frame]["valence"], audio_frames[audio_frame]["arousal"])

        if reset_frame_counter > 10:
            #generate new video frame
            print("change video frame due to timeout")
            last_category = audio_frames[audio_frame]["category"]
            video_frames[frame_index] = audio_frames[audio_frame]
            if frame_index != 0:
                video_frames[frame_index-1]["endTime"] = video_frames[frame_index]["startTime"]
            temp_energy = audio_frames[audio_frame]["rms_mean"]
            if energy_mean != 0:
                video_frames[frame_index]["rms_mean"] = energy_mean
            energy_mean = temp_energy
            frame_index += 1
            reset_frame_counter = 0
        elif last_category != audio_frames[audio_frame]["category"]:
            #generate new video frame
            last_category = audio_frames[audio_frame]["category"]
            video_frames[frame_index] = audio_frames[audio_frame]
            if frame_index != 0:
                video_frames[frame_index-1]["endTime"] = video_frames[frame_index]["startTime"]
            temp_energy = audio_frames[audio_frame]["rms_mean"]
            if energy_mean != 0:
                video_frames[frame_index]["rms_mean"] = energy_mean
            energy_mean = temp_energy
            frame_index += 1
            print("change video frame due to category change")
            reset_frame_counter = 0
        
        elif math.fabs(energy_mean - audio_frames[audio_frame]["rms_mean"]) > audio_frames[audio_frame]["rms_std"]*2:
            #generate new video frame
            last_category = audio_frames[audio_frame]["category"]
            video_frames[frame_index] = audio_frames[audio_frame]
            if frame_index != 0:
                video_frames[frame_index-1]["endTime"] = video_frames[frame_index]["startTime"]
            temp_energy = audio_frames[audio_frame]["rms_mean"]
            if energy_mean != 0:
                video_frames[frame_index]["rms_mean"] = energy_mean
            energy_mean = temp_energy
            frame_index += 1
            print("change video frame due to large change in energy")
            reset_frame_counter = 0
        else:
            # keep current video frame and update mean and std value
            if reset_frame_counter != 0:
                energy_mean = energy_mean * (reset_frame_counter-1)/reset_frame_counter + audio_frames[audio_frame]["rms_mean"] / reset_frame_counter
            else:
                energy_mean = audio_frames[audio_frame]["rms_mean"]

        reset_frame_counter += 1

    video_frames[frame_index-1]["endTime"] = song_data["song_length"]
    for video_frame in video_frames:
        video_frames[video_frame]["frame_count"] = (video_frames[video_frame]["endTime"] - video_frames[video_frame]["startTime"]) * 24

    new_file = f"scripts/{output_script_name}.jwfscript"
    header_file = "scriptTemplates/header.txt"
    footer_file = "scriptTemplates/footer.txt"
    print(video_frames[0])
    with open(new_file, 'w') as outfile:
        with open(header_file, 'r') as infile:
            content = infile.read()
            outfile.write(f"// begin header section\n")
            outfile.write(content)
            outfile.write(f"\n// end header section\n")

        outfile.write(f"\n")

        for idx, video_frame in enumerate(video_frames):
            outfile.write(f'AddFlameMoviePart(flameMovie, {video_frames[video_frame]["category"]}({idx},{video_frames[video_frame]["valence"] * 3}, {video_frames[video_frame]["arousal"] * -2}, {video_frames[video_frame]["color_start"][0]}, {video_frames[video_frame]["color_start"][1]}, {video_frames[video_frame]["color_start"][2]}, {video_frames[video_frame]["color_end"][0]}, {video_frames[video_frame]["color_end"][1]}, {video_frames[video_frame]["color_end"][2]}, {video_frames[video_frame]["rms_mean"]*10}), {int(video_frames[video_frame]["frame_count"])});')
            outfile.write("\n")
        outfile.write(f'flameMovie.getGlobalScripts()[0] = new GlobalScript(GlobalScriptType.ROTATE_ROLL, {song_data["bpm"]});')
        with open(footer_file, 'r') as infile:
            content = infile.read()
            outfile.write(f"\n// begin footer section\n")
            outfile.write(content)
            outfile.write(f"\n// end footer section")

        with open(f"scriptTemplates/mandelbrot_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start linear only gen flame\n")
            outfile.write(content)
            outfile.write("\n// end linear only gen flame\n")

        with open(f"scriptTemplates/linear_only_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start linear only gen flame\n")
            outfile.write(content)
            outfile.write("\n// end linear only gen flame\n")

        with open(f"scriptTemplates/brokat_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start brokat_gen flame\n")
            outfile.write(content)
            outfile.write("\n// end brokat_gen flame\n")

        with open(f"scriptTemplates/galaxies_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start galaxies_gen flame\n")
            outfile.write(content)
            outfile.write("\n// end galaxies_gen flame\n")

        with open(f"scriptTemplates/julian_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start julian_gen flame\n")
            outfile.write(content)
            outfile.write("\n// end julian_gen flame\n")

        with open(f"scriptTemplates/synth_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n// start synth_gen flame\n")
            outfile.write(content)
            outfile.write("\n// end synth_gen flame\n")

        with open(f"scriptTemplates/palette_gen.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n")
            outfile.write(content)

        with open(f"scriptTemplates/add_flame_movie_part.txt", 'r') as infile:
            content = infile.read()
            outfile.write("\n")
            outfile.write(content)
    

    
    #write_script(template_file, new_file, placeholders)


# arousal_file_path = "Data/deam/arousal.csv"
# arousal_df = pd.read_csv(arousal_file_path)
# valence_file_path = "Data/deam/valence.csv"
# valence_df = pd.read_csv(valence_file_path)
# 
# audio_features_df = pd.read_csv("Data/deam/audio_features/2.csv", delimiter=";")
# mask = (audio_features_df.index >= 30) & (audio_features_df.index <= 89)
# audio_features_df = audio_features_df[mask]
# #print(audio_features_df)
# test = np.append(audio_features_df.columns.values, ["valence", "arousal"])
# 
# 
# 
# #audio_features_df = audio_features_df.to_dict()
# #print(audio_features)
# #test_af = audio_features[(audio_features[:, 0] >= 15) & (audio_features[:, 0] < 45)]
# #print(af)
# #audio_features = audio_features_df.values
# #audio_features = audio_features_df.to_numpy()
# #print(test_af)
# test_v = valence_df.head(1).to_numpy()[0]
# test_a = arousal_df.head(1).to_numpy()[0]
# # test_v = valence_df.head(2).to_numpy()[1]
# # test_a = arousal_df.head(2).to_numpy()[1]
# 
# test_a = test_a[~np.isnan(test_a)]
# test_a = test_a[1::]
# test_v = test_v[1::]
# test_v = test_v[~np.isnan(test_v)]
# 
# #af = dict.fromkeys(test[1::])
# #print(af)
#end_time = 0
# test_af = {}
# row = audio_features_df.iloc[0]
# #print(len(audio_features_df))
# for i in range(len(audio_features_df)):
#     row = audio_features_df.iloc[i]
#     test_af[row["frameTime"]] = {
#         "pcm_RMSenergy_sma_amean": row["pcm_RMSenergy_sma_amean"],
#         "pcm_RMSenergy_sma_stddev": row["pcm_RMSenergy_sma_stddev"],
#         "valence": test_v[i],
#         "arousal": test_a[i]
#     }
#     if i == len(audio_features_df)-1:
#         end_time = row["frameTime"] + 0.5

# print(test_af)
# print(end_time)


# feature extraction

def load_data(audio_file_path, sample_rate, output_script_name, guess_bpm):
    y, sr = librosa.load(audio_file_path, sr=sample_rate)
    y = librosa.resample(y=y, orig_sr=sr, target_sr=44100)
    audio_features = extract_features(y, 44100, 0.5).to_numpy()
    genre_features = extract_features(y, 44100, len(y)//44100).to_numpy()[:,1:]
    bpm_value = extract_bpm(y, 44100, guess_bpm)
    print(bpm_value)
#print(bpm_value)
#print(audio_features)
#print(genre_features)
# still need bpm

# detect genre and emotion
    valence_model = joblib.load('trained_models/valence_rf_model.pkl')
    arousal_model = joblib.load('trained_models/arousal_rf_model.pkl')
    genre_model = joblib.load('trained_models/genre_rf_model.pkl')

    valence_values = valence_model.predict(audio_features[:,1:])
#print(valence_value)
    arousal_values = arousal_model.predict(audio_features[:,1:])
    genre_value = genre_model.predict(genre_features)
    genre_value = ["rock"]
    print(genre_value)
# values, counts = np.unique(genre_values, return_counts=True)
# max_count_index = np.argmax(counts)
#print(genre_values)
# print(values[max_count_index])
# build script
#print(audio_features)
    song_data = {
        "genre": genre_value[0],
        "audio_frames":{},
        "bpm": bpm_value[0]/120
    }

#print(audio_features[0][28])
    end_time = 0
    for i in range(len(audio_features)):
        song_data["audio_frames"][audio_features[i][0]] = {
            "valence": valence_values[i],
            "arousal": arousal_values[i],
            "rms_mean": audio_features[i][27],
            "rms_std": audio_features[i][28]
        }
        if i == len(audio_features)-1:
            end_time = audio_features[i][0] + 0.5

    song_data["song_length"] = end_time
#print(end_time)
# GET BPM
#print(song_data)
    generate_script(song_data, output_script_name)
# 
y, sr = librosa.load("audio/rock_30_secs.mp3", sr=44100)
#y = librosa.resample(y=y, orig_sr=sr, target_sr=44100)
audio_features = extract_features(y, 44100, 0.5).to_numpy()
#genre_features = extract_features(y, 44100, len(y)//44100).to_numpy()[:,1:]
#bpm_value = extract_bpm(y, 44100)
# #print(bpm_value)
# #print(audio_features)
# #print(genre_features)
# # still need bpm
# 
# # detect genre and emotion
valence_model = joblib.load('trained_models/valence_rf_model.pkl')
arousal_model = joblib.load('trained_models/arousal_rf_model.pkl')
# genre_model = joblib.load('trained_models/genre_rf_model.pkl')
# 
valence_values = valence_model.predict(audio_features[:,1:])
# #print(valence_value)
arousal_values = arousal_model.predict(audio_features[:,1:])
v = np.mean(valence_values)
a = np.mean(arousal_values)
# print(v)
# print(a)
#print(v*3)
#print(a*-2)
def get_paper_color(x,y):
    img = Image.open("tt.png") #tt
    width, height = img.size

    pixel_x = int((x + 1) / 2 * width)
    pixel_y = int((1 - (y + 1) / 2) * height)
    return img.getpixel((pixel_x, pixel_y))
color1 = get_paper_color(v,a)
#print(color1) 
angle = math.atan2(a, v)
color2 = get_paper_color(math.cos(angle)*85/255, math.sin(angle)*85/255)
color3 = get_paper_color(math.cos(angle), math.sin(angle))
#print(color2)
#print(color3)
# genre_value = genre_model.predict(genre_features)
# # values, counts = np.unique(genre_values, return_counts=True)
# # max_count_index = np.argmax(counts)
# #print(genre_values)
# # print(values[max_count_index])
# # build script
# #print(audio_features)
# song_data = {
#     "genre": genre_value[0],
#     "audio_frames":{},
#     "bpm": bpm_value[0]/120
# }
# 
# #print(audio_features[0][28])
# end_time = 0
# for i in range(len(audio_features)):
#     song_data["audio_frames"][audio_features[i][0]] = {
#         "valence": valence_values[i],
#         "arousal": arousal_values[i],
#         "rms_mean": audio_features[i][27],
#         "rms_std": audio_features[i][28]
#     }
#     if i == len(audio_features)-1:
#         end_time = audio_features[i][0] + 0.5
# 
# song_data["song_length"] = end_time
# #print(end_time)
# # GET BPM
# #print(song_data)
# generate_script(song_data)
#angles = np.degrees(np.arctan2(test_a[0::], test_v[0::])) % 360

#angle = math.degrees(math.atan2(test_a[0], test_v[0]))


#generate_script(test_af)
#print(get_color(-0.27, -0.068))

load_data("audio/rock_30_secs.mp3", 44100, "rock_sample", 120)
#load_data("Data/audio_files/genre_set/jazz.00000.wav", 22050, "jazz_sample")
#load_data("Data/audio_files/genre_set/metal.00001.wav", 22050, "metal_sample")
#load_data("Data/audio_files/genre_set/pop.00001.wav", 22050, "pop_sample")
#load_data("output_30sec.mp3", 44100, "classical_sample")
#load_data("Data/audio_files/genre_set/rock.00004.wav", 22050, "rock_sample")

#1,2,3,4
#5,6,7,8,9
#10,11,12,13,14,15,16,17,18,19
#20,21,22,23,24,25,26,27,28,29
#30,31,32,33,34,35,36,37,38,39
#40,41,42,43,44,45,46,47,48,49
#50,51,52,53,54,55,56,57,58,59
#60,61,62,63,64