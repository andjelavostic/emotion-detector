import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# putanja do RAVDESS dataset-a
DATA_PATH = "../data/raw"

# emocije iz dataset-a
emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# folder gde će se čuvati slike
os.makedirs("eda_photos", exist_ok=True)

# lista za praćenje da ne dupliraš slike
seen = set()

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split('-')
            if len(parts) < 7:
                continue

            emotion = emotion_labels.get(parts[2])
            actor_id = int(parts[6].split('.')[0])
            gender = 'female' if actor_id % 2 == 0 else 'male'


            key = f"{gender}_{emotion}"
            if key in seen:
                continue  # već imamo tu kombinaciju

            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path)

            # waveplot
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f"Waveform - {gender.capitalize()} - {emotion}")
            plt.tight_layout()
            plt.savefig(f"eda_photos/{key}_wave.png")
            plt.close()

            # mel spectrogram
            plt.figure(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.title(f"Mel Spectrogram - {gender.capitalize()} - {emotion}")
            plt.tight_layout()
            plt.savefig(f"eda_photos/{key}_melspec.png")
            plt.close()

            seen.add(key)

print("✅ Završeno! Sve slike su sačuvane u folderu 'plots'.")
