import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# =======================
# LOAD DATA
# =======================
X_train = np.load("../data/processed_data/X_train.npy")
y_train = np.load("../data/processed_data/y_train.npy")
X_val   = np.load("../data/processed_data/X_val.npy")
y_val   = np.load("../data/processed_data/y_val.npy")
X_test  = np.load("../data/processed_data/X_test.npy")
y_test  = np.load("../data/processed_data/y_test.npy")

# =======================
# LABEL ENCODING
# =======================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)


# =======================
# TRAIN GAUSSIAN NAIVE BAYES
# =======================
gnb = GaussianNB()
gnb.fit(X_train, y_train_enc)

# =======================
# EVALUATE ON VALIDATION SET
# =======================
y_val_pred = gnb.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred))
print("\nClassification Report:\n")
print(classification_report(y_val_enc, y_val_pred, target_names=le.classes_))

# =======================
# EVALUATE ON TEST SET
# =======================
y_test_pred = gnb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred))
print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_))

# =======================
# CONFUSION MATRIX
# =======================
cm = confusion_matrix(y_test_enc, y_test_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
os.makedirs("../models/naive_bayes", exist_ok=True)
plt.savefig("../models/naive_bayes/confusion_matrix.png")
plt.show()
print("Confusion matrix saved as PNG.")

# =======================
# SAVE MODEL, SCALER, LABEL ENCODER
# =======================
joblib.dump(gnb, "../models/naive_bayes/gnb_model.pkl")
joblib.dump(le, "../models/naive_bayes/label_encoder.pkl")
print("Gaussian Naive Bayes model, scaler, and label encoder saved successfully.")
