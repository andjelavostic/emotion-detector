import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# CONFIG
DATASET_PATH = "../data/raw"  # path to folder with RAVDESS
SAMPLE_RATE = 22050           # standard sample rate
N_MFCC = 13                   # number of MFCC coefficients
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# AUGMENTATION FUNCTIONS
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def my_time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def my_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_volume(y, factor=1.2):
    return y * factor

# FEATURE EXTRACTION
def extract_features(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0
    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.hstack([mfccs, pitch_mean, energy, zcr])

# PARSE EMOTION
def parse_emotion_from_filename(filename):
    # Name example: 03-01-05-01-02-01-12.wav
    parts = filename.split('-')
    modality = int(parts[0])
    vocal_channel = int(parts[1])
    emotion = int(parts[2])

    #  (vocal_channel == 1)
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

# LOADING AND AUGMENTATION
X = []
y = []

print("Loading and extracting features with augmentation...")

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

        file_path = os.path.join(actor_path, file)
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Original features
        X.append(extract_features(y_audio, sr))
        y.append(emotion_label)

        # Augmentations
        X.append(extract_features(add_noise(y_audio), sr))
        y.append(emotion_label)

        X.append(extract_features(my_time_stretch(y_audio, 1.1), sr))
        y.append(emotion_label)

        X.append(extract_features(my_pitch_shift(y_audio, sr, 2), sr))
        y.append(emotion_label)

        X.append(extract_features(change_volume(y_audio, 1.2), sr))
        y.append(emotion_label)

X = np.array(X)
y = np.array(y)
print(f"Total examples after augmentation: {len(X)}")

# TRAIN/VAL/TEST SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# SAVE .NPY FILES
os.makedirs("../data/processed_data", exist_ok=True)
np.save("../data/processed_data/X_train.npy", X_train)
np.save("../data/processed_data/y_train.npy", y_train)
np.save("../data/processed_data/X_val.npy", X_val)
np.save("../data/processed_data/y_val.npy", y_val)
np.save("../data/processed_data/X_test.npy", X_test)
np.save("../data/processed_data/y_test.npy", y_test)

print("Saved all .npy files in 'data/processed_data/'")
