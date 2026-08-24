import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

csv_path = "../data/processed_data/audio_labels.csv"
output_dir = "../data/processed_data/cnn/"
os.makedirs(output_dir, exist_ok=True)

duration = 3
sr = 44100
n_mels = 128
hop_length = 512

df = pd.read_csv(csv_path)

features, labels, actors = [], [], []
max_frames = 0

for path, label, actor in tqdm(zip(df['path'], df['emotion'], df['actor']), total=len(df)):
    y, _ = librosa.load(path, sr=sr, duration=duration, offset=0.5, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel).T
    features.append(mel_db.astype(np.float32))
    labels.append(label)
    actors.append(actor)
    max_frames = max(max_frames, mel_db.shape[0])

features_padded = []
for f in features:
    if f.shape[0] < max_frames:
        f = np.pad(f, ((0, max_frames - f.shape[0]), (0, 0)), mode='constant')
    else:
        f = f[:max_frames, :]
    features_padded.append(f)

X_all = np.array(features_padded, dtype=np.float32)
y_all = np.array(labels)
actors_all = np.array(actors)

test_actors = [21, 22, 23, 24]
val_actors  = [17, 18, 19, 20]

train_idx = np.where(~np.isin(actors_all, test_actors + val_actors))[0]
val_idx   = np.where(np.isin(actors_all, val_actors))[0]
test_idx  = np.where(np.isin(actors_all, test_actors))[0]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val,   y_val   = X_all[val_idx],   y_all[val_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

n_mels_dim = X_train.shape[2]
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, n_mels_dim))

def apply_scaler(X):
    shape = X.shape
    return scaler.transform(X.reshape(-1, n_mels_dim)).reshape(shape)

X_train = apply_scaler(X_train)
X_val   = apply_scaler(X_val)
X_test  = apply_scaler(X_test)

joblib.dump(scaler, os.path.join(output_dir, "cnn_scaler.pkl"))

np.save(os.path.join(output_dir, "X_train_cnn.npy"), X_train)
np.save(os.path.join(output_dir, "y_train_cnn.npy"), y_train)
np.save(os.path.join(output_dir, "X_val_cnn.npy"), X_val)
np.save(os.path.join(output_dir, "y_val_cnn.npy"), y_val)
np.save(os.path.join(output_dir, "X_test_cnn.npy"), X_test)
np.save(os.path.join(output_dir, "y_test_cnn.npy"), y_test)

df.iloc[train_idx][['path', 'emotion']].to_csv(os.path.join(output_dir, "train_cnn_paths.csv"), index=False)

print("Gotovo - podela po glumcima, bez curenja podataka")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")