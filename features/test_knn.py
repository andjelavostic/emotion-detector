# eval_knn.py
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

knn = joblib.load("../models/knn/knn_model.pkl")
scaler = joblib.load("../models/knn/scaler.pkl")
le = joblib.load("../models/knn/label_encoder.pkl")

X_test  = np.load("../data/processed_data/knn_and_nb/X_test.npy")
y_test  = np.load("../data/processed_data/knn_and_nb/y_test.npy")

X_test_scaled = scaler.transform(X_test)

y_test_pred = knn.predict(X_test_scaled)

y_test_str = le.inverse_transform(y_test)
y_test_pred_str = le.inverse_transform(y_test_pred)

print("Test Accuracy:", accuracy_score(y_test_str, y_test_pred_str))
print(classification_report(y_test_str, y_test_pred_str, target_names=le.classes_))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Test Confusion Matrix")
plt.show()
