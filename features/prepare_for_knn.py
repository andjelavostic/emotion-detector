import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# ===============================
# SETTINGS
# ===============================
train_csv = "../data/processed_data/knn_and_nb/features_mel_train_augmented.csv"
val_csv   = "../data/processed_data/knn_and_nb/features_mel_val.csv"
test_csv  = "../data/processed_data/knn_and_nb/features_mel_test.csv"

output_dir = "../data/processed_data/knn_and_nb/"

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

X_val   = val_df.drop(columns=['emotion','gender','actor','path']).values
y_val   = val_df['emotion'].values

X_test  = test_df.drop(columns=['emotion','gender','actor','path']).values
y_test  = test_df['emotion'].values

# ===============================
# LABEL ENCODING
# ===============================
le = LabelEncoder()
le.fit(y_train)
y_test_encoded = le.transform(y_test)

# ===============================
# SPAJANJE TRAIN + VAL
# GridSearchCV (u train_knn.py i train_naive_bayes.py) ima sopstvenu unutrasnju
# unakrsnu proveru za biranje hiperparametara, pa train i val spajamo u jedan
# trening skup vec ovde. Skaliranje se radi posebno u svakoj trening skripti,
# na ovom spojenom skupu.
# ===============================
X_trainval = np.concatenate([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
y_trainval_encoded = le.transform(y_trainval)

# ===============================
# SAVE NUMPY ARRAYS + LABEL ENCODER
# ===============================
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_trainval.npy"), X_trainval)
np.save(os.path.join(output_dir, "y_trainval.npy"), y_trainval_encoded)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test_encoded)

joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

print("Train+Val i Test skupovi sacuvani u .npy")
print(f"Train+Val: {X_trainval.shape}, Test: {X_test.shape}")
