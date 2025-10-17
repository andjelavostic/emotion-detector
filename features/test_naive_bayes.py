import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD TEST DATA
# =========================
X_test  = np.load("../data/processed_data/knn_and_nb/X_test.npy")
y_test  = np.load("../data/processed_data/knn_and_nb/y_test.npy")

# =========================
# LOAD SAVED MODEL AND TRANSFORMERS
# =========================
nb = joblib.load("../models/naive_bayes/naive_bayes_model.pkl")
scaler = joblib.load("../models/naive_bayes/scaler.pkl")
le = joblib.load("../models/naive_bayes/label_encoder.pkl")
pca = joblib.load("../models/naive_bayes/pca.pkl")  # Ako si PCA sačuvao

# =========================
# TRANSFORM TEST DATA
# =========================
X_test_trans = scaler.transform(X_test)
X_test_trans = pca.transform(X_test_trans)  # PCA transformacija

# =========================
# PREDICTIONS
# =========================
y_test_pred = nb.predict(X_test_trans)

# =========================
# DECODE LABELS
# =========================
y_test_str = le.inverse_transform(y_test)
y_test_pred_str = le.inverse_transform(y_test_pred)

# =========================
# ACCURACY & REPORT
# =========================
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
    plt.show()  # Ovaj red prikazuje matricu odmah
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #plt.savefig(filename)
    plt.close()
    print(f"✅ Saved {filename}")


save_confusion_matrix(y_test, y_test_pred, le.classes_, "../models/naive_bayes/test_confusion_matrix.png", "Test Confusion Matrix")
