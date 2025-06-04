
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import joblib


def prepare_data(targets):
    X = []
    y = []

    for target in targets:
        target = target[~np.isnan(target)]

        song_id = target[0]
        audio_features_df = pd.read_csv(f"Data/audio_features/emotion_set/{int(song_id)}.csv", delimiter=",")
        audio_features_arr = audio_features_df.iloc[30:,1:].to_numpy()
        #print(audio_features_arr)
        if len(target) != len(audio_features_arr):
            min_len = min(len(target), len(audio_features_arr))
            audio_features_arr = audio_features_arr[:min_len]
            target = target[:min_len]
        for i in range(len(target)):
            if i == 0:
                continue

            X.append(audio_features_arr[i])
            y.append(target[i])
    
    
    return X,y

def display_metrics(y_true, y_pred, target_title):
    rmse = root_mean_squared_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    rrse = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    print(f"Mesurements for {target_title}")
    print(f"r (Correlation): {r:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RAE: {rae:.4f}")
    print(f"RRSE: {rrse:.4f}")
    print("\n")
def train_data(data, target_title):
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_metrics(y_pred = y_pred, y_true=y_test, target_title=target_title)
    joblib.dump(model, f'trained_models/{target_title}_rf_model.pkl')



arousal_file_path = "Data/deam/arousal.csv"
arousal_df = pd.read_csv(arousal_file_path)
valence_file_path = "Data/deam/valence.csv"
valence_df = pd.read_csv(valence_file_path)
mask = (valence_df.index >= 1744) #& (valence_df.index <= 1801)
valence_df = valence_df[mask]
arousal_df = arousal_df[mask]
#print(valence_df)

test_v = valence_df.to_numpy()[:,:]
test_a = arousal_df.to_numpy()[:,:]
train_data(test_v, "valence")
train_data(test_a, "arousal")

