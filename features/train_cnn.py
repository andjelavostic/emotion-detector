import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ===== CONFIG =====
DATA_DIR = "/kaggle/input/heeeeej/processed_data"
MODEL_DIR = "kaggle/working/novo"
BATCH_SIZE = 32
LR = 0.0005
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===== DATASET =====
class MelDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MelDataset(os.path.join(DATA_DIR, "X_train_mel_aug.npy"),
                           os.path.join(DATA_DIR, "y_train_mel_aug.npy"))
val_dataset = MelDataset(os.path.join(DATA_DIR, "X_val_mel.npy"),
                         os.path.join(DATA_DIR, "y_val_mel.npy"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== CNN MODEL =====
class CNNEmotion(nn.Module):
    def __init__(self, n_classes):
        super(CNNEmotion, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (64//8) * (128//8), 128),  # primer
            nn.ReLU(),
            nn.Dropout(0.3),  # gasi 30% neurona
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

n_classes = len(np.load(os.path.join(DATA_DIR, "label_classes_mel.npy")))
model = CNNEmotion(n_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)

# ===== TRAIN LOOP =====
best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []

for epoch in range(EPOCHS):
    model.train()
    y_true_train, y_pred_train = [], []
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        y_true_train.extend(y_batch.cpu().numpy())
        y_pred_train.extend(outputs.argmax(dim=1).cpu().numpy())

    train_loss = running_loss / len(train_dataset)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_f1 = f1_score(y_true_train, y_pred_train, average='weighted')

    # VALIDATION
    model.eval()
    y_true_val, y_pred_val = [], []
    val_running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_running_loss += loss.item() * X_batch.size(0)
            y_true_val.extend(y_batch.cpu().numpy())
            y_pred_val.extend(outputs.argmax(dim=1).cpu().numpy())

    val_loss = val_running_loss / len(val_dataset)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_cnn_model.pth"))

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

print("Training finished. Best model saved.")

# ===== PLOT AND SAVE GRAPHS =====
os.makedirs(MODEL_DIR, exist_ok=True)

# Loss plot
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))
plt.close()

# Accuracy plot
plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "accuracy_plot.png"))
plt.close()

# F1 score plot
plt.figure(figsize=(8,6))
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Val F1")
plt.title("F1 Score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "f1_plot.png"))
plt.close()
