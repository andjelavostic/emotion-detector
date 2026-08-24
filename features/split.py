import pandas as pd
import os

input_csv = "../data/processed_data/knn_and_nb/features_mel_extended.csv"
output_dir = "../data/processed_data/knn_and_nb/"

df = pd.read_csv(input_csv)

test_actors = [21, 22, 23, 24]
val_actors  = [17, 18, 19, 20]
train_actors = [a for a in range(1, 25) if a not in test_actors + val_actors]

train_df = df[df['actor'].isin(train_actors)]
val_df   = df[df['actor'].isin(val_actors)]
test_df  = df[df['actor'].isin(test_actors)]

os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, "features_mel_train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "features_mel_val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "features_mel_test.csv"), index=False)

print("Sacuvano - podela po glumcima")
print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")