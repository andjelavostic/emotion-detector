import os
import inspect
import numpy as np
import pandas as pd
import librosa
import joblib
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ===============================
# SETTINGS
# ===============================
csv_path = "../data/processed_data/audio_labels.csv"
output_dir = "../models/wav2vec2"
os.makedirs(output_dir, exist_ok=True)

WAV2VEC2_CHECKPOINT = "facebook/wav2vec2-base"
TARGET_SR = 16000
CLIP_DURATION = 3
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ista podela po glumcu kao kod KNN/NB/CNN - test glumci se nikad ne pojavljuju
# u treningu ni u biranju hiperparametara, radi poštenog poređenja modela.
TEST_ACTORS = [21, 22, 23, 24]
VAL_ACTORS = [17, 18, 19, 20]
TRAIN_ACTORS = [a for a in range(1, 25) if a not in TEST_ACTORS + VAL_ACTORS]

# ===============================
# AUGMENTACIJA (samo za train, primenjuje se nasumicno po uzorku)
# ===============================
def add_noise(data, noise_factor=0.005):
    return data + noise_factor * np.random.randn(len(data))

def time_shift(data, shift_max=0.2):
    shift = np.random.randint(int(len(data) * -shift_max), int(len(data) * shift_max))
    return np.roll(data, shift)

# ===============================
# UCITAVANJE LABELA I PODELA PO GLUMCU
# ===============================
df_labels = pd.read_csv(csv_path)

train_paths = df_labels[df_labels['actor'].isin(TRAIN_ACTORS)].reset_index(drop=True)
val_paths = df_labels[df_labels['actor'].isin(VAL_ACTORS)].reset_index(drop=True)
test_paths = df_labels[df_labels['actor'].isin(TEST_ACTORS)].reset_index(drop=True)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# ===============================
# DATASET (radi direktno na sirovom audio signalu, bez mel-spektrograma)
# ===============================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_CHECKPOINT)

le = LabelEncoder()
le.fit(EMOTIONS)

class RavdessAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_encoder, augment=False):
        self.paths = df['path'].tolist()
        self.labels = label_encoder.transform(df['emotion'].tolist())
        self.augment = augment
        self.target_len = TARGET_SR * CLIP_DURATION

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=TARGET_SR, duration=CLIP_DURATION, offset=0.5)
        if self.augment:
            choice = np.random.choice(["none", "noise", "shift"])
            if choice == "noise":
                y = add_noise(y)
            elif choice == "shift":
                y = time_shift(y)
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]
        inputs = feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="np")
        return {
            "input_values": inputs["input_values"][0].astype(np.float32),
            "labels": int(self.labels[idx]),
        }

train_dataset = RavdessAudioDataset(train_paths, le, augment=True)
val_dataset = RavdessAudioDataset(val_paths, le, augment=False)
test_dataset = RavdessAudioDataset(test_paths, le, augment=False)

# ===============================
# MODEL (transfer learning - fine-tuning pretreniranog Wav2Vec2)
# ===============================
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    WAV2VEC2_CHECKPOINT,
    num_labels=len(le.classes_),
    label2id={l: i for i, l in enumerate(le.classes_)},
    id2label={i: l for i, l in enumerate(le.classes_)},
)
# Zamrzavamo konvolucioni feature-encoder sloj (nizak nivo audio karakteristika, vec
# dobro naucen na Librispeech-u) - fine-tune-ujemo samo transformer + klasifikacionu glavu.
model.freeze_feature_encoder()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_EPOCHS = 10
steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = max(1, int(0.1 * total_steps))

# Naziv parametra za strategiju evaluacije se razlikuje izmedju verzija
# biblioteke transformers (evaluation_strategy / eval_strategy), pa se
# proverava koji naziv prihvata instalirana verzija.
ta_params = inspect.signature(TrainingArguments.__init__).parameters
eval_strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=3e-5,
    warmup_steps=warmup_steps,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    report_to="none",
    **{eval_strategy_key: "epoch"},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ===============================
# EVALUACIJA NA TEST SKUPU
# ===============================
test_results = trainer.predict(test_dataset)
y_pred = np.argmax(test_results.predictions, axis=1)
y_true = test_results.label_ids

test_acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {test_acc * 100:.2f}%")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ===============================
# CONFUSION MATRIX
# ===============================
def save_confusion_matrix(y_true, y_pred, classes, filename, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

save_confusion_matrix(
    y_true, y_pred, le.classes_,
    os.path.join(output_dir, "test_confusion_matrix.png"),
    "Wav2Vec2 - Test Confusion Matrix",
)

# ===============================
# CUVANJE MODELA
# ===============================
trainer.save_model(os.path.join(output_dir, "final_model"))
feature_extractor.save_pretrained(os.path.join(output_dir, "final_model"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

print(f"Model i rezultati sacuvani u {output_dir}")
