import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ===============================
# SETTINGS
# ===============================
csv_path = "../data/processed_data/audio_labels.csv"
output_dir = "../data/processed_data/cnn/"
os.makedirs(output_dir, exist_ok=True)

duration = 3      # seconds
sr = 44100
n_mels = 128
hop_length = 512

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(csv_path)

# ===============================
# EXTRACT MEL SPECTROGRAMS
# ===============================
features = []
labels = []
max_frames = 0

for path, label in tqdm(zip(df['path'], df['emotion']), total=len(df), desc="Extracting features"):
    y, _ = librosa.load(path, sr=sr, duration=duration, offset=0.5, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel)
    mel_db = mel_db.T  # (frames, n_mels)
    
    features.append(mel_db.astype(np.float32))
    labels.append(label)
    max_frames = max(max_frames, mel_db.shape[0])

# ===============================
# PAD / TRIM
# ===============================
features_padded = []
for f in features:
    if f.shape[0] < max_frames:
        f_padded = np.pad(f, ((0,max_frames-f.shape[0]),(0,0)), mode='constant')
    else:
        f_padded = f[:max_frames, :]
    features_padded.append(f_padded)

X_all = np.array(features_padded, dtype=np.float32)
y_all = np.array(labels)

# ===============================
# STANDARDIZE po mel bins
# ===============================
n_samples, n_frames, n_mels = X_all.shape
X_all = X_all.reshape(-1, n_mels)
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)
X_all = X_all.reshape(n_samples, n_frames, n_mels)

# SAVE scaler
joblib.dump(scaler, os.path.join(output_dir, "cnn_scaler.pkl"))

# ===============================
# SPLIT 70/15/15
# ===============================
train_idx, temp_idx = train_test_split(np.arange(len(y_all)), test_size=0.3, stratify=y_all, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y_all[temp_idx], random_state=42)

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_val   = X_all[val_idx]
y_val   = y_all[val_idx]
X_test  = X_all[test_idx]
y_test  = y_all[test_idx]

# ===============================
# SAVE NPYS
# ===============================
np.save(os.path.join(output_dir, "X_train_cnn.npy"), X_train)
np.save(os.path.join(output_dir, "y_train_cnn.npy"), y_train)
np.save(os.path.join(output_dir, "X_val_cnn.npy"), X_val)
np.save(os.path.join(output_dir, "y_val_cnn.npy"), y_val)
np.save(os.path.join(output_dir, "X_test_cnn.npy"), X_test)
np.save(os.path.join(output_dir, "y_test_cnn.npy"), y_test)

# ===============================
# SAVE train CSV sa putanjama (za augmentaciju kasnije)
# ===============================
train_csv_path = os.path.join(output_dir, "train_cnn_paths.csv")
df.iloc[train_idx][['path','emotion']].to_csv(train_csv_path, index=False)

# ===============================
# PRINT INFO
# ===============================
print("✅ Feature extraction, standardization and split done!")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(f"✅ Train CSV for augmentation saved: {train_csv_path}")
print(f"✅ Scaler saved: {os.path.join(output_dir, 'cnn_scaler.pkl')}")
