import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# ===============================
# SETTINGS
# ===============================
train_csv = "../data/processed_data/features_mel_train_augmented.csv"
val_csv   = "../data/processed_data/features_mel_val.csv"
test_csv  = "../data/processed_data/features_mel_test.csv"

output_dir = "../data/processed_data/"

# ===============================
# LOAD CSVs
# ===============================
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# ===============================
# FEATURES & LABELS
# ===============================
X_train = train_df.drop(columns=['emotion','gender']).values
y_train = train_df['emotion'].values

X_val   = val_df.drop(columns=['emotion','gender','path']).values
y_val   = val_df['emotion'].values

X_test  = test_df.drop(columns=['emotion','gender','path']).values
y_test  = test_df['emotion'].values

# ===============================
# LABEL ENCODING
# ===============================
le = LabelEncoder()
le.fit(y_train)  # fit samo na train, da izbegnemo "future info"
y_train_encoded = le.transform(y_train)
y_val_encoded   = le.transform(y_val)
y_test_encoded  = le.transform(y_test)

# ===============================
# STANDARDIZE
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
X_val_scaled   = scaler.transform(np.nan_to_num(X_val, nan=0.0))
X_test_scaled  = scaler.transform(np.nan_to_num(X_test, nan=0.0))

# ===============================
# SAVE NUMPY ARRAYS
# ===============================
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir,"X_train.npy"), X_train_scaled)
np.save(os.path.join(output_dir,"X_val.npy"), X_val_scaled)
np.save(os.path.join(output_dir,"X_test.npy"), X_test_scaled)

np.save(os.path.join(output_dir,"y_train.npy"), y_train_encoded)
np.save(os.path.join(output_dir,"y_val.npy"), y_val_encoded)
np.save(os.path.join(output_dir,"y_test.npy"), y_test_encoded)

# ===============================
print("✅ Train/Val/Test datasets saved to .npy")
print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
