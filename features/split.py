import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ===============================
# SETTINGS
# ===============================
#input_csv = "../data/processed_data/features_mel.csv"
input_csv = "../data/processed_data/features_mel_extended.csv"
output_dir = "../data/processed_data/"

# Load features CSV
df = pd.read_csv(input_csv)

# Stratified split 70/15/15
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['emotion'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['emotion'])

# Save CSVs
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, "features_mel_train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "features_mel_val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "features_mel_test.csv"), index=False)

print("✅ Saved train/val/test CSVs")
print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
