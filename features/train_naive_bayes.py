import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ============================
# LOAD DATA
# ============================
X_train = np.load("../data/processed_data/X_train.npy")
y_train = np.load("../data/processed_data/y_train.npy")
X_val   = np.load("../data/processed_data/X_val.npy")
y_val   = np.load("../data/processed_data/y_val.npy")

# ============================
# LABEL ENCODING
# ============================
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)

# ============================
# PREPROCESSING FOR NAIVE BAYES
# ============================
# 1️⃣ Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# 2️⃣ Power transform to approximate normal distribution
pt = PowerTransformer(method='yeo-johnson')
X_train_trans = pt.fit_transform(X_train_scaled)
X_val_trans   = pt.transform(X_val_scaled)

# ============================
# TRAIN GAUSSIAN NAIVE BAYES MODEL
# ============================
nb = GaussianNB()
nb.fit(X_train_trans, y_train_encoded)

# ============================
# VALIDATION
# ============================
y_val_pred_encoded = nb.predict(X_val_trans)
y_val_pred_str = le.inverse_transform(y_val_pred_encoded)
y_val_str = y_val

print("Validation Accuracy:", accuracy_score(y_val_str, y_val_pred_str))
print(classification_report(y_val_str, y_val_pred_str, target_names=le.classes_))

# ============================
# SAVE MODEL + TRANSFORMERS
# ============================
os.makedirs("../models/naive_bayes", exist_ok=True)
joblib.dump(nb, "../models/naive_bayes/naive_bayes_model.pkl")
joblib.dump(scaler, "../models/naive_bayes/scaler.pkl")
joblib.dump(pt, "../models/naive_bayes/power_transformer.pkl")
joblib.dump(le, "../models/naive_bayes/label_encoder.pkl")

print("Naive Bayes model and preprocessing tools saved successfully.")
