import os
import random
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# CONFIG
DATA_DIR = "../data/raw"  # folder with Actor_01, Actor_02...
SAMPLE_RATE = 22050
N_MELS = 64
MAX_LEN = 128  # time of frame numbers
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Emotion mapping
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}


# LOAD FILES AND EXTRACT MEL-SPECTROGRAMS
X = []
y = []

for actor_folder in os.listdir(DATA_DIR):
    actor_path = os.path.join(DATA_DIR, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(actor_path, file)
        parts = file.split("-")
        emotion_code = parts[2]
        channel_code = parts[1]

        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            continue

        y_audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)

        # Fall or truncate
        if mel_db.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]

        X.append(mel_db)
        y.append(emotion)

# Convert list to numpy array BEFORE adding channel dimension
X = np.array(X)
X = X[:, np.newaxis, :, :]  # (N, 1, n_mels, max_len)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/val/test split 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=TEST_SIZE + VAL_SIZE, stratify=y_encoded, random_state=42,shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
    stratify=y_temp, random_state=42,shuffle=True
)

# Save processed data
os.makedirs("../data/processed_data", exist_ok=True)
np.save("../data/processed_data/X_train_mel.npy", X_train)
np.save("../data/processed_data/X_val_mel.npy", X_val)
np.save("../data/processed_data/X_test_mel.npy", X_test)
np.save("../data/processed_data/y_train_mel.npy", y_train)
np.save("../data/processed_data/y_val_mel.npy", y_val)
np.save("../data/processed_data/y_test_mel.npy", y_test)
np.save("../data/processed_data/label_classes_mel.npy", le.classes_)

print("Preprocessing done. Mel-spektrogram dataset saved.")
