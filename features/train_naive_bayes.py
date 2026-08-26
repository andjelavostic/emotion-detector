import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
X_trainval = np.load("../data/processed_data/knn_and_nb/X_trainval.npy")
y_trainval = np.load("../data/processed_data/knn_and_nb/y_trainval.npy")
X_test = np.load("../data/processed_data/knn_and_nb/X_test.npy")
y_test = np.load("../data/processed_data/knn_and_nb/y_test.npy")

le = joblib.load("../data/processed_data/knn_and_nb/label_encoder.pkl")

# =========================
# STANDARDIZACIJA (fit na train+val)
# =========================
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(np.nan_to_num(X_trainval, nan=0.0))
X_test_scaled = scaler.transform(np.nan_to_num(X_test, nan=0.0))

# =========================
# GRID SEARCH ZA PCA + GAUSSIAN NAIVE BAYES
# Broj PCA komponenti se bira preko GridSearchCV (sopstvena unutrasnja unakrsna
# provera nad train+val skupom), umesto da se unapred fiksira.
# =========================
nb_pipeline = Pipeline([('pca', PCA()), ('nb', GaussianNB())])
param_grid = {'pca__n_components': [20, 50, 100, 150]}
grid = GridSearchCV(nb_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_trainval_scaled, y_trainval)

nb_final = grid.best_estimator_
print("Best Naive Bayes params (PCA n_components):", grid.best_params_)

# =========================
# EVALUACIJA NA TEST SKUPU
# =========================
y_test_pred = nb_final.predict(X_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# =========================
# SAVE CONFUSION MATRIX
# =========================
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

save_confusion_matrix(y_test, y_test_pred, le.classes_, "../models/naive_bayes/test_confusion_matrix.png", "Test Confusion Matrix")

# =========================
# SAVE MODEL + TRANSFORMERS
# =========================
os.makedirs("../models/naive_bayes", exist_ok=True)
joblib.dump(nb_final, "../models/naive_bayes/naive_bayes_model.pkl")
joblib.dump(scaler, "../models/naive_bayes/scaler.pkl")
joblib.dump(le, "../models/naive_bayes/label_encoder.pkl")
print("Naive Bayes model, scaler, label encoder and confusion matrix saved successfully.")
