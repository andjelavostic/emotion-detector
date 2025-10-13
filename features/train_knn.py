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

# Encode labels to integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded   = le.transform(y_val)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# KNN model with hyperparameter search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1=Manhattan, 2=Euclidean
}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train_encoded)  # fit on encoded labels

# Best model
print("Best KNN parameters found:", grid.best_params_)
best_knn = grid.best_estimator_

# Evaluate on validation set
y_val_pred = best_knn.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val_encoded, y_val_pred))
print(classification_report(y_val_encoded, y_val_pred, target_names=le.classes_))

# Save model
joblib.dump(best_knn, "../data/processed_data/knn_model.pkl")
print("KNN model saved successfully.")

