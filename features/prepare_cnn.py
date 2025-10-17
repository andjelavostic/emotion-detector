import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Učitaj CSV-ove
train_df = pd.read_csv("../data/processed_data/train_cnn.csv")
val_df = pd.read_csv("../data/processed_data/val_cnn.csv")
test_df = pd.read_csv("../data/processed_data/test_cnn.csv")

# Odvoji X i y
y_train = train_df['emotion']
X_train = train_df.drop(columns=['emotion', 'gender', 'path']).values
y_val = val_df['emotion']
X_val = val_df.drop(columns=['emotion', 'gender', 'path']).values
y_test = test_df['emotion']
X_test = test_df.drop(columns=['emotion', 'gender', 'path']).values

# Normalizacija
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_val=(X_val-mean)/std
X_test = (X_test - mean)/std

# Reshape u 2D oblik za CNN
#X_train = X_train.reshape(-1, 128, 130, 1)
#X_val = X_val.reshape(-1, 128, 130, 1)
#X_test = X_test.reshape(-1, 128, 130, 1)
X_train = X_train[:,:,np.newaxis]
X_val = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

# Label encoding
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_val_enc = to_categorical(le.transform(y_val))
y_test_enc = to_categorical(le.transform(y_test))

print("✅ CNN input shapes:")
print("Train:", X_train.shape, y_train_enc.shape)
print("Val:", X_val.shape, y_val_enc.shape)
print("Test:", X_test.shape, y_test_enc.shape)
