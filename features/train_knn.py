import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Load train and validation data
X_train = np.load("../data/processed_data/X_train.npy")
y_train = np.load("../data/processed_data/y_train.npy")
X_val   = np.load("../data/processed_data/X_val.npy")
y_val   = np.load("../data/processed_data/y_val.npy")

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Grid search with more hyperparameters
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40]
}

grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train_encoded)

# Best model
best_knn = grid.best_estimator_
print("Best KNN parameters found:", grid.best_params_)

# Evaluate on validation set
y_val_pred_encoded = best_knn.predict(X_val_scaled)
y_val_pred_str = le.inverse_transform(y_val_pred_encoded)  # dekodiraj predikcije
y_val_str = y_val  # stvarne vrednosti već u string formatu

print("Validation Accuracy:", accuracy_score(y_val_str, y_val_pred_str))
print(classification_report(y_val_str, y_val_pred_str, target_names=le.classes_))

# Save model, scaler, and label encoder
joblib.dump(best_knn, "../models/knn/knn_model.pkl")
joblib.dump(scaler, "../models/knn/scaler.pkl")
joblib.dump(le, "../models/knn/label_encoder.pkl")
print("KNN model, scaler, and label encoder saved successfully.")
