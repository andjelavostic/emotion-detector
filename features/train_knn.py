import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === LOAD DATA ===
X_train = np.load("../data/processed_data/knn_and_nb/X_train.npy")
y_train = np.load("../data/processed_data/knn_and_nb/y_train.npy")  # integeri
X_val   = np.load("../data/processed_data/knn_and_nb/X_val.npy")
y_val   = np.load("../data/processed_data/knn_and_nb/y_val.npy")
X_test  = np.load("../data/processed_data/knn_and_nb/X_test.npy")
y_test  = np.load("../data/processed_data/knn_and_nb/y_test.npy")

# === STANDARDIZE FEATURES ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# === ENCODE LABELS ===
le = LabelEncoder()
# Ako su y_train već integeri 0..7, ovo samo osigurava da LabelEncoder ima imena emocija
le.fit(['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])

# === GRID SEARCH FOR KNN ===
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3,5,7,9,11],
    'weights': ['uniform','distance'],
    'p':[1,2]
    #'algorithm':['auto','ball_tree','kd_tree','brute'],
    #'leaf_size':[20,30,40]
}
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train)

best_knn = grid.best_estimator_
print("Best KNN parameters:", grid.best_params_)

# === PREDICTIONS ===
y_val_pred = best_knn.predict(X_val_scaled)
y_test_pred = best_knn.predict(X_test_scaled)

# === DECODE LABELS ZA ISPIS ===
y_val_str  = le.inverse_transform(y_val)
y_val_pred_str = le.inverse_transform(y_val_pred)
y_test_str = le.inverse_transform(y_test)
y_test_pred_str = le.inverse_transform(y_test_pred)

# === ACCURACY & REPORT ===
print("Validation Accuracy:", accuracy_score(y_val_str, y_val_pred_str))
print(classification_report(y_val_str, y_val_pred_str, target_names=le.classes_))

print("Test Accuracy:", accuracy_score(y_test_str, y_test_pred_str))
print(classification_report(y_test_str, y_test_pred_str, target_names=le.classes_))

# === SAVE CONFUSION MATRIX ===
def save_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved {filename}")

save_confusion_matrix(y_val, y_val_pred, le.classes_, "../models/knn/val_confusion_matrix.png", "Validation Confusion Matrix")
save_confusion_matrix(y_test, y_test_pred, le.classes_, "../models/knn/test_confusion_matrix.png", "Test Confusion Matrix")

# === SAVE MODEL, SCALER, LABEL ENCODER ===
os.makedirs("../models/knn", exist_ok=True)
joblib.dump(best_knn, "../models/knn/knn_model.pkl")
joblib.dump(scaler, "../models/knn/scaler.pkl")
joblib.dump(le, "../models/knn/label_encoder.pkl")
print("KNN model, scaler, label encoder and confusion matrices saved successfully.")
