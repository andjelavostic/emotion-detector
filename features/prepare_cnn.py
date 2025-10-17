import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os

# ===============================
# 1️⃣ Učitaj CSV-ove
# ===============================
val_df   = pd.read_csv("../data/processed_data/cnn/val_cnn.csv")
test_df  = pd.read_csv("../data/processed_data/cnn/test_cnn.csv")

# ===============================
# 2️⃣ Odvoji X i y
# ===============================
y_train = train_df['emotion']
X_train = train_df.drop(columns=['emotion', 'gender']).values

y_val = val_df['emotion']
X_val = val_df.drop(columns=['emotion', 'gender', 'path']).values

y_test = test_df['emotion']
X_test = test_df.drop(columns=['emotion', 'gender', 'path']).values

# ===============================
# 3️⃣ Normalizacija (sa zaštitom)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ===============================
# 4️⃣ Reshape u 2D oblik za 1D CNN
# ===============================
X_train = X_train[:, :, np.newaxis]  # shape: (samples, timesteps, features=1)
X_val   = X_val[:, :, np.newaxis]
X_test  = X_test[:, :, np.newaxis]

# ===============================
# 5️⃣ Label encoding i one-hot
# ===============================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# ===============================
# 6️⃣ Sačuvaj npy fajlove
# ===============================
output_dir = "../data/processed_data/cnn/"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train_cnn.npy"), X_train)
np.save(os.path.join(output_dir, "y_train_cnn.npy"), y_train_enc)
np.save(os.path.join(output_dir, "X_val_cnn.npy"), X_val)
np.save(os.path.join(output_dir, "y_val_cnn.npy"), y_val_enc)
np.save(os.path.join(output_dir, "X_test_cnn.npy"), X_test)
np.save(os.path.join(output_dir, "y_test_cnn.npy"), y_test_enc)

# ===============================
# 7️⃣ Provera
# ===============================
print("✅ CNN input shapes:")
print("Train:", X_train.shape, y_train_enc.shape)
print("Val:  ", X_val.shape, y_val_enc.shape)
print("Test: ", X_test.shape, y_test_enc.shape)
print("✅ .npy fajlovi su sačuvani u", output_dir)
