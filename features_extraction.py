import librosa
import numpy as np
import pandas as pd
import re
import os

def load_audio(file_path, sr=44100):
    y, _ = librosa.load(file_path, sr=sr)
    return y


# def extract_features(y, sr, hop_length=22050):  # 500ms = 22050 samples at 44.1kHz
#     timestamps = librosa.frames_to_time(range(len(y) // hop_length), sr=sr, hop_length=hop_length)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
#     rms = librosa.feature.rms(y=y, hop_length=hop_length)
#     rms_delta = librosa.feature.delta(rms)[0]
#     rms_delta_std = np.std(rms_delta)
#     print(rms_delta)
#     print(rms_delta_std)
# 
#     zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
# 
#     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#     beat_times = librosa.frames_to_time(beat_frames, sr=sr)
#     #print(f"Tempo: {tempo} BPM")
#     #print(f"Beat Times: {beat_times}")
#     feature_matrix = np.vstack([mfcc, rms, zcr, spectral_centroid])
#     min_len = min(feature_matrix.shape[1], len(timestamps))
#     feature_matrix = feature_matrix[:, :min_len]
#     timestamps = timestamps[:min_len]
#     feature_df = pd.DataFrame(feature_matrix.T, columns=[f"mfcc_{i}" for i in range(13)] + ["rms", "zcr", "spectral_centroid"])
#     feature_df.insert(0, 'timestamp', timestamps[:feature_df.shape[0]])
# 
#     return feature_df

def extract_bpm(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

def extract_features(y, sr, window_duration):
    frame_length = int(0.060 * sr)  # 60ms
    hop_length = int(0.010 * sr)    # 10ms hop

    # Fine-grained feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length, frame_length=frame_length)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)

    # All features dictionary
    features = {
        "mfcc": mfcc,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spec_centroid,
        "chroma_stft": chroma_stft,
        "spec_bandwidth": spec_bandwidth,
        "spec_contrast": spec_contrast,
        "spec_rolloff": spec_rolloff
    }
    
    n_frames = min(f.shape[1] for f in features.values())
    #print(n_frames)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    #window_duration = window_duration  # seconds
    window_size_frames = int(window_duration / 0.010)  # 0.5s / 10ms = 50 frames
    #print(frame_times)
    total_duration = frame_times[-1]
    #print(total_duration)
    n_windows = int(np.floor(total_duration / window_duration))

    all_rows = []

    for i in range(n_windows):
        start_time = i * window_duration
        end_time = (i + 1) * window_duration
        #print(end_time)

        # Find frames within this 500ms window
        mask = (frame_times >= start_time) & (frame_times < end_time)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue  # skip empty windows

        window_data = {}
        window_data['timestamp'] = round(start_time, 3)  # clean timestamp (0.0, 0.5, 1.0, etc.)

        for name, feat in features.items():
            if name == "mfcc":
                for j in range(13):
                    window_data[f'mfcc_{j}_mean'] = feat[j, idx].mean()
                    window_data[f'mfcc_{j}_std'] = feat[j, idx].std()
            else:
                window_data[f'{name}_mean'] = feat[0, idx].mean()
                window_data[f'{name}_std'] = feat[0, idx].std()

        all_rows.append(window_data)

    # Build final DataFrame
    feature_df = pd.DataFrame(all_rows)
    return feature_df



import soundfile as sf
file_path = "Data/audio_files/genre_set/jazz.00054.wav"
def is_valid_audio(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return True
    except (RuntimeError, sf.LibsndfileError):
        return False
# if is_valid_audio(file_path):
#     y = load_audio(file_path)
# else:
#     print(f"Invalid or corrupted audio file: {file_path}")
#     y = None

# try:
#     y = load_audio("Data/audio_files/genre_set/jazz.00054.wav")
#     y = librosa.resample(y=y, orig_sr=22050, target_sr=44100)
#     features_df = extract_features(y, sr=44100, window_duration=len(y)//44100)
#     features_df.to_csv('output.csv', index=False)
# except(librosa.util.exceptions.ParameterError, FileNotFoundError, ValueError) as e:
#     print("error")
#     y = None
# y = librosa.resample(y=y, orig_sr=22050, target_sr=44100)
# features_df = extract_features(y, sr=44100, window_duration=len(y)//44100)
#import soundfile as sf

# sf.write('resampled_audio.wav', y, 44100)
#print(features_df)
#features_df.to_csv('output.csv', index=False)



# folder_path = "Data/audio_files/emotion_set"
# csv_files = sorted(
#     [f for f in os.listdir(folder_path) if f.endswith('.mp3')],
#     key=lambda x: int(re.findall(r'\d+', x)[0])
# )
# 
# # print(csv_files)
# # 
# for csv_file in csv_files:
#     y = load_audio(f"Data/audio_files/emotion_set/{csv_file}")
#     features_df = extract_features(y, sr=44100, window_duration=0.5)
#     features_df.to_csv(f'Data/audio_features/emotion_set/{csv_file[:4]}.csv', index=False)
# 

# folder_path = "Data/audio_files/genre_set"
# csv_files = sorted(
#     [f for f in os.listdir(folder_path) if f.endswith('.wav')],
#     key=lambda x: int(re.findall(r'\d+', x)[0])
# )
# 
# print(csv_files)
# # 
# for csv_file in csv_files:
#     #print(csv_file)
#     if not is_valid_audio(f"Data/audio_files/genre_set/{csv_file}"):
#         print(f"Invalid or corrupted audio file: {csv_file}")
#         continue
#         
#     y = load_audio(f"Data/audio_files/genre_set/{csv_file}", 22050)
#     y = librosa.resample(y=y, orig_sr=22050, target_sr=44100)
#     features_df = extract_features(y, sr=44100, window_duration=len(y)//44100)
#     features_df.to_csv(f'Data/audio_features/genre_set/{csv_file[:-4]}.csv', index=False)