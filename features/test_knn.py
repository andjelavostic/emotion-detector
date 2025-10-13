import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model, scaler, encoder
knn = joblib.load("../models/knn/knn_model.pkl")
scaler = joblib.load("../models/knn/scaler.pkl")
le = joblib.load("../models/knn/label_encoder.pkl")

# Load test data
X_test = np.load("../data/processed_data/X_test.npy")
y_test = np.load("../data/processed_data/y_test.npy")  # string labels

# Scale features
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred_encoded = knn.predict(X_test_scaled)
y_pred = le.inverse_transform(y_pred_encoded)  # decode to original labels

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("../models/knn/confusion_matrix.png")
print("Confusion matrix saved as PNG.")