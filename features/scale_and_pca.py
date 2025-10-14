from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import joblib
import os

# Load raw features
X_train = np.load("../data/processed_data/X_train.npy")
X_val   = np.load("../data/processed_data/X_val.npy")
X_test  = np.load("../data/processed_data/X_test.npy")

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# PCA (opcionalno)
pca = PCA(0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Save preprocessed features i scaler/PCA
os.makedirs("../data/preprocessed", exist_ok=True)
np.save("../data/preprocessed/X_train_pca.npy", X_train_pca)
np.save("../data/preprocessed/X_val_pca.npy", X_val_pca)
np.save("../data/preprocessed/X_test_pca.npy", X_test_pca)
joblib.dump(scaler, "../data/preprocessed/scaler.pkl")
joblib.dump(pca, "../data/preprocessed/pca.pkl")
