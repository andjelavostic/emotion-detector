import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
DATASET_PATH = "../data/raw"
SAMPLE_RATE = 22050
N_MFCC = 40
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ===== AUGMENTATION FUNCTIONS =====
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def my_time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def my_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_volume(y, factor=1.2):
    return y * factor

# ===== FEATURE EXTRACTION =====
def extract_from_signal(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.hstack([mfccs, pitch_mean, energy, zcr])

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return extract_from_signal(y, sr)

# ===== PARSE EMOTION =====
def parse_emotion_from_filename(filename):
    parts = filename.split('-')
    if len(parts) < 3:
        return None
    vocal_channel = int(parts[1])
    emotion = int(parts[2])
    if vocal_channel != 1:
        return None
    emotions = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotions.get(emotion, None)

# ===== LOAD FILE PATHS AND LABELS =====
file_paths = []
labels = []

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        emotion_label = parse_emotion_from_filename(file)
        if emotion_label is None:
            continue

        file_paths.append(os.path.join(actor_path, file))
        labels.append(emotion_label)

file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"Found {len(file_paths)} audio files with labels.")

# ===== SPLIT DATA =====
train_paths, temp_paths, y_train, temp_labels = train_test_split(
    file_paths, labels, test_size=TEST_SIZE + VAL_SIZE, stratify=labels, random_state=42
)

val_paths, test_paths, y_val, y_test = train_test_split(
    temp_paths, temp_labels, 
    test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
    stratify=temp_labels,
    random_state=42
)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# ===== EXTRACT FEATURES + AUGMENT TRAIN =====
X_train = []
y_train_aug = []

print("Processing and augmenting train set...")
for file_path, label in zip(train_paths, y_train):
    y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # original
    X_train.append(extract_from_signal(y_audio, sr))
    y_train_aug.append(label)

    # augmentations
    X_train.append(extract_from_signal(add_noise(y_audio), sr))
    X_train.append(extract_from_signal(my_time_stretch(y_audio, 1.1), sr))
    X_train.append(extract_from_signal(my_pitch_shift(y_audio, sr, 2), sr))
    X_train.append(extract_from_signal(change_volume(y_audio, 1.2), sr))
    y_train_aug += [label] * 4

X_train = np.array(X_train)
y_train_aug = np.array(y_train_aug)

print(f"Train set after augmentation: {len(X_train)} samples")

# ===== EXTRACT FEATURES FOR VAL AND TEST =====
def extract_features_from_paths(paths):
    X = []
    for file_path in paths:
        X.append(extract_features(file_path))
    return np.array(X)

print("Processing validation set...")
X_val = extract_features_from_paths(val_paths)
print("Processing test set...")
X_test = extract_features_from_paths(test_paths)

# ===== SAVE TO .NPY =====
os.makedirs("../data/processed_data", exist_ok=True)

np.save("../data/processed_data/X_train.npy", X_train)
np.save("../data/processed_data/y_train.npy", y_train_aug)
np.save("../data/processed_data/X_val.npy", X_val)
np.save("../data/processed_data/y_val.npy", y_val)
np.save("../data/processed_data/X_test.npy", X_test)
np.save("../data/processed_data/y_test.npy", y_test)

print("All .npy files saved in 'data/processed_data/'")
