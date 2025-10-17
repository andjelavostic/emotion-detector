import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ===============================
# SETTINGS
# ===============================
train_csv = "../data/processed_data/knn_and_nb/features_mel_train.csv"  # samo train CSV
output_dir = "../data/processed_data/knn_and_nb"
train_aug_csv = os.path.join(output_dir,"features_mel_train_augmented.csv")
duration = 3
sr = 44100
offset = 0.5

# ===============================
# AUDIO AUGMENTATION FUNCTIONS
# ===============================
def add_noise(data, noise_factor=0.005):
    return data + noise_factor * np.random.randn(len(data))

def time_shift(data, shift_max=0.2):
    shift = np.random.randint(int(len(data) * -shift_max), int(len(data) * shift_max))
    return np.roll(data, shift)

def stretch(data, rate=None):
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def dynamic_change(data):
    factor = np.random.uniform(0.7, 1.3)
    return data * factor

augment_funcs = [
    lambda y, sr: y,           # original
    lambda y, sr: add_noise(y),
    lambda y, sr: time_shift(y),
    lambda y, sr: stretch(y),
    lambda y, sr: pitch_shift(y, sr),
    lambda y, sr: dynamic_change(y)
]


# ===============================
# LOAD TRAIN CSV
# ===============================
df_train = pd.read_csv(train_csv)

# ===============================
# AUGMENT TRAIN ONLY
# ===============================
all_features = []
all_emotions = []
all_genders = []

for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Augmenting TRAIN"):
    y_path = row['path']
    emotion = row['emotion']
    gender = row['gender']
    
    # učitaj audio
    y, _ = librosa.load(y_path, duration=duration, sr=sr, offset=offset, res_type='kaiser_fast')
    
    for func in augment_funcs:
        y_aug = func(y, sr)
        
        # Mel spektrogram
        mel = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel)
        
        # očisti NaN i inf vrednosti
        mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Izračunaj statistike po vremenskoj osi (axis=1)
        mean_features = np.mean(mel_db, axis=1)
        std_features  = np.std(mel_db, axis=1)
        max_features  = np.max(mel_db, axis=1)
            
        # spoji u jedan feature vector
        feature_vector = np.concatenate([mean_features,std_features,max_features])
        
        # dodaj u listu
        all_features.append(feature_vector)
        all_emotions.append(emotion)
        all_genders.append(gender)
# ===============================
# SAVE AUGMENTED TRAIN CSV
# ===============================
df_aug_train = pd.DataFrame(all_features)
df_aug_train['emotion'] = all_emotions
df_aug_train['gender'] = all_genders

# Reorder columns: emotion, gender, features
cols = ['emotion', 'gender'] + [c for c in df_aug_train.columns if c not in ['emotion', 'gender']]
df_aug_train = df_aug_train[cols]

os.makedirs(output_dir, exist_ok=True)
df_aug_train.to_csv(train_aug_csv, index=False)
print(f"✅ Augmented TRAIN dataset saved: {df_aug_train.shape} -> {train_aug_csv}")
