import numpy as np
import librosa
import joblib
import gradio as gr
from tensorflow.keras.models import load_model

# --- Konstante (moraju biti ISTE kao u feature_extraction_cnn.py i train_cnn.py) ---
SR = 44100
DURATION = 3
OFFSET = 0.5
N_MELS = 128
HOP_LENGTH = 512
FMAX = 8000
MAX_FRAMES = 259  # dužina sekvence na kojoj je model treniran
EMOTIONS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # alfabetski - isti redosled kao LabelEncoder

# --- Učitavanje sačuvanog modela i skalera ---
model = load_model("models/cnn/cnn_model.h5")
scaler = joblib.load("data/processed_data/cnn/cnn_scaler.pkl")

def extract_features(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, duration=DURATION, offset=OFFSET, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel).T  # (vreme, 128)

    if mel_db.shape[0] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, MAX_FRAMES - mel_db.shape[0]), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:MAX_FRAMES, :]

    mel_scaled = scaler.transform(mel_db)
    return mel_scaled[np.newaxis, :, :, np.newaxis]  # (1, 259, 128, 1)

def predict_emotion(audio_path):
    if audio_path is None:
        return "Snimi ili učitaj audio zapis."
    X = extract_features(audio_path)
    probs = model.predict(X)[0]
    return {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Govorni zapis"),
    outputs=gr.Label(num_top_classes=8, label="Prepoznata emocija"),
    title="Prepoznavanje emocija iz govora",
    description="Učitaj ili snimi kratak govorni zapis (do 3 sekunde) — model (1D CNN) predviđa emociju."
)

demo.launch()