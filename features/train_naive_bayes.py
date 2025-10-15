import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
# =========================
# LOAD DATA
# =========================
X_train = np.load("../data/processed_data/X_train.npy")
y_train = np.load("../data/processed_data/y_train.npy")  # integeri 0..7
X_val   = np.load("../data/processed_data/X_val.npy")
y_val   = np.load("../data/processed_data/y_val.npy")
X_test  = np.load("../data/processed_data/X_test.npy")
y_test  = np.load("../data/processed_data/y_test.npy")

# =========================
# STANDARDIZE + POWER TRANSFORM
# =========================
scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_val_trans  = scaler.transform(X_val)
X_test_trans  = scaler.transform(X_test)

pca = PCA(n_components=50)
X_train_trans = pca.fit_transform(X_train_trans)
X_val_trans   = pca.transform(X_val_trans)
X_test_trans=pca.transform(X_test_trans)

# =========================
# LABEL ENCODER (samo za imena emocija)
# =========================
le = LabelEncoder()
le.fit(['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])

# =========================
# TRAIN GAUSSIAN NAIVE BAYES
# =========================
nb = GaussianNB()
nb.fit(X_train_trans, y_train)

# =========================
# PREDICTIONS
# =========================
y_val_pred = nb.predict(X_val_trans)
y_test_pred = nb.predict(X_test_trans)

# =========================
# DECODE LABELS ZA ISPIS
# =========================
y_val_str  = le.inverse_transform(y_val)
y_val_pred_str = le.inverse_transform(y_val_pred)
y_test_str = le.inverse_transform(y_test)
y_test_pred_str = le.inverse_transform(y_test_pred)

# =========================
# ACCURACY & REPORT
# =========================
print("Validation Accuracy:", accuracy_score(y_val_str, y_val_pred_str))
print(classification_report(y_val_str, y_val_pred_str, target_names=le.classes_))

print("Test Accuracy:", accuracy_score(y_test_str, y_test_pred_str))
print(classification_report(y_test_str, y_test_pred_str, target_names=le.classes_))

# =========================
# SAVE CONFUSION MATRIX
# =========================
def save_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved {filename}")

save_confusion_matrix(y_val, y_val_pred, le.classes_, "../models/naive_bayes/val_confusion_matrix.png", "Validation Confusion Matrix")
save_confusion_matrix(y_test, y_test_pred, le.classes_, "../models/naive_bayes/test_confusion_matrix.png", "Test Confusion Matrix")

# =========================
# SAVE MODEL + TRANSFORMERS
# =========================
os.makedirs("../models/naive_bayes", exist_ok=True)
joblib.dump(nb, "../models/naive_bayes/naive_bayes_model.pkl")
joblib.dump(scaler, "../models/naive_bayes/scaler.pkl")
#joblib.dump(pt, "../models/naive_bayes/power_transformer.pkl")
joblib.dump(le, "../models/naive_bayes/label_encoder.pkl")
print("Naive Bayes model, scaler, power transformer, label encoder and confusion matrices saved successfully.")
