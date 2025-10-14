import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# LOAD TEST DATA
# ============================
X_test = np.load("../data/processed_data/X_test.npy")
y_test = np.load("../data/processed_data/y_test.npy")

# ============================
# LOAD MODEL AND PREPROCESSORS
# ============================
nb = joblib.load("../models/naive_bayes/naive_bayes_model.pkl")
scaler = joblib.load("../models/naive_bayes/scaler.pkl")
pt = joblib.load("../models/naive_bayes/power_transformer.pkl")
le = joblib.load("../models/naive_bayes/label_encoder.pkl")

# ============================
# PREPROCESS TEST DATA
# ============================
X_test_scaled = scaler.transform(X_test)
X_test_trans = pt.transform(X_test_scaled)

# ============================
# PREDICT
# ============================
y_test_pred_encoded = nb.predict(X_test_trans)
y_test_pred_str = le.inverse_transform(y_test_pred_encoded)

# ============================
# EVALUATE
# ============================
print("🔹 Test Accuracy:", accuracy_score(y_test, y_test_pred_str))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_str, target_names=le.classes_))

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_test, y_test_pred_str, labels=le.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Naive Bayes (Test Set)")
plt.show()
