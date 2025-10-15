import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../data/processed_data/audio_labels.csv")

features = []

for path in tqdm(df['path'], desc="Extracting features"):
    y, sr = librosa.load(path, duration=3, sr=44100, offset=0.5, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel)
    mean_features = np.mean(mel_db, axis=1)
    features.append(mean_features)

feature_df = pd.DataFrame(features)
final_df = pd.concat([df[['emotion', 'gender','path']], feature_df], axis=1)
final_df.to_csv("../data/processed_data/features_mel.csv", index=False)


print("✅ Sačuvan feature dataframe:", final_df.shape)
