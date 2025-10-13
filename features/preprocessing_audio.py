import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# CONFIG
DATASET_PATH = "../data/raw"   # path
SAMPLE_RATE = 22050                # standard for RAVDESS
N_MFCC = 13                        # MFFC coeficcients
TEST_SIZE = 0.15                   # 15% for test
VAL_SIZE = 0.15                    # 15% from the rest for validation

# FUNCTIONS

def extract_features(file_path):
    """Extracting MFCC, pitch, energy i ZCR."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # MFCC (mean value per coef)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)

    # Pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes) else 0

    # Energy (Root Mean Square)
    energy = np.mean(librosa.feature.rms(y=y))

    # ZCR (Zero Crossing Rate)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # All in one vector
    features = np.hstack([mfccs, pitch_mean, energy, zcr])
    return features


def parse_emotion_from_filename(filename):
    """Extracting emotion from file name"""
    # Name of file example : 03-01-05-01-02-01-12.wav
    parts = filename.split('-')
    modality = int(parts[0])
    vocal_channel = int(parts[1])
    emotion = int(parts[2])

    #  (vocal_channel == 1)
    if vocal_channel != 1:
        return None

    # Emotion classes - 8
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


# LOADING AND PREPROCCESSING
X = []
y = []

print("Loading and extracting...")

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        emotion_label = parse_emotion_from_filename(file)
        if emotion_label is None:
            continue  # skip if it does not contain emotion label

        file_path = os.path.join(actor_path, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(emotion_label)

X = np.array(X)
y = np.array(y)

print(f"Loaded: {len(X)} examples.")


# CREATING TRAIN, TEST AND VAL by spliting
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# SAVE .npy FILES
os.makedirs("../data/processed_data", exist_ok=True)

# Save files
np.save("../data/processed_data/X_train.npy", X_train)
np.save("../data/processed_data/y_train.npy", y_train)
np.save("../data/processed_data/X_val.npy", X_val)
np.save("../data/processed_data/y_val.npy", y_val)
np.save("../data/processed_data/X_test.npy", X_test)
np.save("../data/processed_data/y_test.npy", y_test)


print("Saved .npy files in folder 'data/processed_data/'")
