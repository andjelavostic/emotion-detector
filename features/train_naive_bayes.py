import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# =======================
# LOAD DATA (već podeljeno)
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
# SCALE FEATURES
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# =======================
# OPTIONAL: SMOTE (ako su klase neuravnotežene)
# =======================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_enc)

print(f"Train set before SMOTE: {len(X_train)}, after SMOTE: {len(X_train_bal)}")

# =======================
# TRAIN GAUSSIANNB
# =======================
model = GaussianNB()
model.fit(X_train_bal, y_train_bal)

# =======================
# VALIDATION
# =======================
y_val_pred = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val_enc, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")
print(classification_report(y_val_enc, y_val_pred, target_names=le.classes_, zero_division=0))

# =======================
# TEST
# =======================
y_test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test_enc, y_test_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_, zero_division=0))

# =======================
# CONFUSION MATRIX
# =======================
cm = confusion_matrix(y_test_enc, y_test_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - GaussianNB")
plt.savefig("../models/naive_bayes/confusion_matrix_GaussianNB.png")
plt.close()

# =======================
# SAVE MODEL, SCALER, LABEL ENCODER
# =======================
os.makedirs("../models/naive_bayes", exist_ok=True)
joblib.dump(model, "../models/naive_bayes/GaussianNB.pkl")
joblib.dump(scaler, "../models/naive_bayes/standard_scaler.pkl")
joblib.dump(le, "../models/naive_bayes/label_encoder.pkl")

print("GaussianNB pipeline completed and saved successfully.")
