import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === LOAD DATA ===
X_trainval = np.load("../data/processed_data/knn_and_nb/X_trainval.npy")
y_trainval = np.load("../data/processed_data/knn_and_nb/y_trainval.npy")
X_test = np.load("../data/processed_data/knn_and_nb/X_test.npy")
y_test = np.load("../data/processed_data/knn_and_nb/y_test.npy")

le = joblib.load("../data/processed_data/knn_and_nb/label_encoder.pkl")

# === STANDARDIZACIJA (fit na train+val) ===
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(np.nan_to_num(X_trainval, nan=0.0))
X_test_scaled = scaler.transform(np.nan_to_num(X_test, nan=0.0))

# === GRID SEARCH ZA KNN ===
# GridSearchCV bira hiperparametre pomocu sopstvene unutrasnje unakrsne provere
# (cv=5) nad train+val skupom, i automatski (refit=True) trenira finalni model
# na celom tom skupu.
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_trainval_scaled, y_trainval)

best_knn = grid.best_estimator_
print("Best KNN parameters:", grid.best_params_)

# === EVALUACIJA NA TEST SKUPU ===
y_test_pred = best_knn.predict(X_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# === SAVE CONFUSION MATRIX ===
def save_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

save_confusion_matrix(y_test, y_test_pred, le.classes_, "../models/knn/test_confusion_matrix.png", "Test Confusion Matrix")

# === SAVE MODEL, SCALER, LABEL ENCODER ===
os.makedirs("../models/knn", exist_ok=True)
joblib.dump(best_knn, "../models/knn/knn_model.pkl")
joblib.dump(scaler, "../models/knn/scaler.pkl")
joblib.dump(le, "../models/knn/label_encoder.pkl")
print("KNN model, scaler, label encoder and confusion matrix saved successfully.")
