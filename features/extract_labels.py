import os
import pandas as pd

audio_dir = "../data/raw"
emotion, gender, actor, path_list = [], [], [], []

for folder in os.listdir(audio_dir):
    if folder.startswith("Actor"):
        for file in os.listdir(os.path.join(audio_dir, folder)):
            parts = file.split('-')
            emo_id = int(parts[2])
            actor_id = int(parts[6].split('.')[0])
            gen = "female" if actor_id % 2 == 0 else "male"

            emotion.append(emo_id)
            gender.append(gen)
            actor.append(actor_id)
            path_list.append(os.path.join(audio_dir, folder, file))

df = pd.DataFrame({
    'emotion': emotion,
    'gender': gender,
    'actor': actor,
    'path': path_list
})

emotion_map = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}
df['emotion'] = df['emotion'].map(emotion_map)

df.to_csv("../data/processed_data/audio_labels.csv", index=False)
print("Sačuvani labeli:", df.shape)


