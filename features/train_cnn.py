import os
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

output_dir = "../models/cnn"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 1️⃣ Učitaj npy fajlove
# ===============================
data_dir = "../data/processed_data/cnn"
X_train = np.load(os.path.join(data_dir, "X_train_cnn.npy"))
y_train = np.load(os.path.join(data_dir, "y_train_cnn.npy"))
X_val   = np.load(os.path.join(data_dir, "X_val_cnn.npy"))
y_val   = np.load(os.path.join(data_dir, "y_val_cnn.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test_cnn.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test_cnn.npy"))

# ===============================
# 2️⃣ One-hot encoding labela
# Fit-ujemo na fiksnom skupu od 8 emocija (a ne samo na y_train), da mapiranje
# uvek bude isto bez obzira na to koje su klase prisutne u konkretnom split-u.
# ===============================
le = LabelEncoder()
le.fit(['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
y_train_int = le.transform(y_train)  # 0..7
y_val_int   = le.transform(y_val)
y_test_int  = le.transform(y_test)

# One-hot encode
num_classes = len(le.classes_)
y_train_enc = to_categorical(y_train_int, num_classes=num_classes)
y_val_enc   = to_categorical(y_val_int, num_classes=num_classes)
y_test_enc  = to_categorical(y_test_int, num_classes=num_classes)

X_train = X_train[:, :, np.newaxis] if X_train.ndim==2 else X_train
X_val   = X_val[:, :, np.newaxis] if X_val.ndim==2 else X_val
X_test  = X_test[:, :, np.newaxis] if X_test.ndim==2 else X_test
# ===============================
# 4️⃣ Definicija 1D CNN modela

"""model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
    Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    Dropout(0.4),
    Conv1D(128, kernel_size=3, activation='relu'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])"""
input_shape = X_train.shape[1:]  # (259, 128)

model = Sequential([
    Conv1D(64, kernel_size=20, activation='relu', input_shape=input_shape),
    Conv1D(128, kernel_size=20, activation='relu', kernel_regularizer=l2(0.01), 
    bias_regularizer=l2(0.01)), 
    MaxPooling1D(pool_size=8), 
    Dropout(0.4),
    Conv1D(128, kernel_size=20, activation='relu'), 
    MaxPooling1D(pool_size=8), 
    Dropout(0.4), 
    GlobalAveragePooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.4), 
    Dense(num_classes, activation='softmax')
])

opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# ===============================
# 5️⃣ Trening
# ===============================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(os.path.join(output_dir, "cnn_model.h5"), monitor='val_loss', save_best_only=True),
]

history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

# EarlyStopping(restore_best_weights=True) već vraća najbolje težine u model,
# ali eksplicitno snimamo i ovde da fajl uvek postoji čak i bez ranog zaustavljanja.
model.save(os.path.join(output_dir, "cnn_model.h5"))

# ===============================
# 6️⃣ Evaluacija na test skupu
# ===============================
test_loss, test_acc = model.evaluate(X_test, y_test_enc)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# ===============================
# 7️⃣ Plotovanje
# ===============================
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_acc  = history.history['accuracy']
val_acc    = history.history['val_accuracy']
epochs = range(1, len(train_loss)+1)

plt.figure(figsize=(6,5))
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
plt.close()

plt.figure(figsize=(6,5))
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_enc, axis=1)

# Kreiraj matricu konfuzije
# VAŽNO: display_labels mora biti le.classes_ (redosled u kom LabelEncoder mapira
# stringove na 0..7), a ne ručno napisan spisak - inače su oznake na matrici pogrešne.
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Emotions")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()
plt.close()

print(f"Model i grafici sačuvani u {output_dir}")