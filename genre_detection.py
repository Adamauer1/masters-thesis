import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split

import joblib

def prepare_data(targets):
    X = []
    y = []
    
    for target in targets:
        
        file_name = target[0]
        if file_name == "jazz.00054.wav":
            continue
        audio_features_df = pd.read_csv(f"Data/audio_features/genre_set/{file_name[:-4]}.csv", delimiter=",")
        
        X.append(audio_features_df.to_numpy()[0][1:])
        y.append(target[59])
        
    print(len(X))
    print(len(y))
    return X, y

def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # 'weighted' for multiclass
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

# Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n")

def train_data(data):
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_metrics(y_pred=y_pred, y_true=y_test)
    joblib.dump(model, f'trained_models/genre_rf_model.pkl')
    
    
genre_file_path = "Data/features_30_sec.csv"
genre_df = pd.read_csv(genre_file_path)
genre_arr = genre_df.to_numpy()

train_data(genre_arr)