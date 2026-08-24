import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Učitaj CSV sa audio fajlovima i labelama
df = pd.read_csv("../data/processed_data/audio_labels.csv")

features = []

for path in tqdm(df['path'], desc="Extracting features"):
    # Učitaj audio
    y, sr = librosa.load(path, duration=3, sr=44100, offset=0.5, res_type='kaiser_fast')
    
    # Napravi Mel spektrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel)
    
    # Očisti NaN i inf vrednosti
    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Izračunaj statistike po vremenskoj osi (axis=1)
    mean_features = np.mean(mel_db, axis=1)
    std_features  = np.std(mel_db, axis=1)
    max_features  = np.max(mel_db, axis=1)
        
    # Spoji u jedan feature vektor
    feature_vector = np.concatenate([mean_features,std_features,max_features])
    
    features.append(feature_vector)

# Napravi DataFrame sa svim feature-ima
feature_df = pd.DataFrame(features)

# Spoji sa labelama
final_df = pd.concat([df[['emotion', 'gender', 'actor', 'path']], feature_df], axis=1)

# Sačuvaj u CSV
final_df.to_csv("../data/processed_data/knn_and_nb/features_mel_extended.csv", index=False)

print("Sačuvan feature dataframe:", final_df.shape)
