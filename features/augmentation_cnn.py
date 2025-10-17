import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

# ===============================
# SETTINGS
# ===============================
train_csv = "../data/processed_data/cnn/train_cnn_paths.csv"  # CSV sa samo train putanjama
val_npy = "../data/processed_data/cnn/X_val_cnn.npy"
test_npy = "../data/processed_data/cnn/X_test_cnn.npy"
output_dir = "../data/processed_data/cnn"
n_mels = 128
sr = 44100
duration = 3
offset = 0.5
hop_length = 512

# Load scaler koji smo ranije sačuvali
scaler = joblib.load(os.path.join(output_dir, "cnn_scaler.pkl"))

# ===============================
# AUGMENTATION FUNCTIONS
# ===============================
def add_noise(y): return y + 0.005 * np.random.randn(len(y))
def time_shift(y): return np.roll(y, np.random.randint(-int(len(y)*0.2), int(len(y)*0.2)))
def stretch(y): return librosa.effects.time_stretch(y, rate=np.random.uniform(0.8,1.2))
def pitch_shift(y, sr): return librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2,2))
def dynamic_change(y): return y * np.random.uniform(0.7,1.3)

augment_funcs = [
    lambda y, sr: y,
    lambda y, sr: add_noise(y),
    lambda y, sr: time_shift(y),
    lambda y, sr: stretch(y),
    lambda y, sr: pitch_shift(y, sr),
    lambda y, sr: dynamic_change(y)
]

# ===============================
# LOAD TRAIN CSV (samo putanje)
# ===============================
df_train = pd.read_csv(train_csv)

# ===============================
# AUGMENT AND EXTRACT FEATURES
# ===============================
temp_features = []
all_emotions = []
max_frames = 0

for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
    y_path = row['path']
    emotion = row['emotion']
    
    y, _ = librosa.load(y_path, sr=sr, duration=duration, offset=offset, res_type='kaiser_fast')
    
    for func in augment_funcs:
        y_aug = func(y, sr)
        
        mel = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel)
        mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)
        
        mel_db = mel_db.T
        temp_features.append(mel_db.astype(np.float32))
        all_emotions.append(emotion)
        max_frames = max(max_frames, mel_db.shape[0])



X_train_aug = np.array(temp_features, dtype=np.float32)
y_train_aug = np.array(all_emotions)

# ===============================
# STANDARDIZE using saved scaler
# ===============================
n_samples, n_frames, n_mels = X_train_aug.shape
X_train_aug = X_train_aug.reshape(-1, n_mels)
X_train_aug = scaler.transform(X_train_aug)
X_train_aug = X_train_aug.reshape(n_samples, n_frames, n_mels)

# ===============================
# SAVE NPYS (overwrite train)
# ===============================
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir,"X_train_cnn.npy"), X_train_aug)
np.save(os.path.join(output_dir,"y_train_cnn.npy"), y_train_aug)

print(f"✅ Augmented train dataset saved: {X_train_aug.shape}, {y_train_aug.shape}")
