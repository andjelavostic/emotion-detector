# Emotion Detector

Sistem za prepoznavanje emocija iz govora, treniran i evaluiran na **RAVDESS** skupu podataka. Upoređena su četiri pristupa klasifikaciji: **KNN**, **Naive Bayes**, **1D CNN** i **Wav2Vec2** (fine-tuning pretreniranog transformera za govor).

---

## Skup podataka

[RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song) — 24 profesionalna glumca (12 muških, 12 ženskih), po 60 govornih snimaka svaki, ukupno 1440 audio zapisa. Svaki snimak nosi jednu od 8 emocija: *neutral, calm, happy, sad, angry, fear, disgust, surprise*.

## Metodologija

**Podela na train/val/test je po glumcu (actor-independent split):** 16 glumaca za trening, 4 za validaciju, 4 za test, bez preklapanja. Isti glumac se nikad ne pojavljuje u više od jednog skupa. Ovo je namerna metodološka odluka — nasumična podela bi dozvolila da se isti glas nađe i u trening i u test skupu, pa bi model delom prepoznavao identitet govornika umesto same emocije. Actor-independent podela meri realnu sposobnost generalizacije na nove, neviđene govornike.

**Augmentacija** (šum, pomeraj u vremenu, time-stretch, pitch-shift, promena jačine zvuka) se primenjuje isključivo na trening skup — validacija i test ostaju neizmenjeni originalni snimci.

**Karakteristike:**
- KNN i Naive Bayes koriste statistike (mean/std/max) Mel-spektrograma svedene na fiksni vektor od 384 vrednosti.
- CNN koristi punu vremensku sekvencu Mel-spektrograma (259 vremenskih okvira × 128 mel-frekvencija).
- Wav2Vec2 radi direktno nad sirovim audio signalom, bez ručno dizajniranih karakteristika.

## Modeli

| Model | Pristup | Biranje hiperparametara |
|---|---|---|
| KNN | `KNeighborsClassifier` | `GridSearchCV` (n_neighbors, weights, p) nad train+val skupom |
| Naive Bayes | PCA + `GaussianNB` | `GridSearchCV` bira broj PCA komponenti nad train+val skupom |
| 1D CNN | Konvoluciona neuronska mreža (Conv1D) | Trening na train skupu, `EarlyStopping` prati val skup |
| Wav2Vec2 | Fine-tuning `facebook/wav2vec2-base` | Zamrznut feature-encoder, fine-tune transformer sloja i klasifikacione glave |

Kod KNN i Naive Bayes modela, finalni model se trenira na train+val skupu zajedno (nakon što su hiperparametri odabrani), a test skup se koristi isključivo za završnu evaluaciju. CNN i Wav2Vec2 zadržavaju odvojen val skup jer im je potreban tokom samog treninga (rano zaustavljanje).

## Rezultati

Tačnost na test skupu (actor-independent, glumci koje model nikad nije video):

| Model | Test accuracy |
|---|---|
| Naive Bayes | 36.3% |
| KNN | 37.9% |
| CNN | 55.8% |
| Wav2Vec2 | 75.0% |

Za poređenje: nasumično pogađanje između 8 klasa daje ~12.5% tačnosti. Actor-independent evaluacija daje niže, ali metodološki pouzdanije brojeve nego nasumična podela na train/val/test, kod koje bi model mogao delom da prepoznaje identitet govornika umesto same emocije.

## Struktura projekta

```
emotion-detector/
├── data/
│   ├── raw/                          RAVDESS audio zapisi (Actor_01 ... Actor_24)
│   └── processed_data/               izvedeni feature-i, podela na skupove, .npy nizovi
├── features/                         skripte za pripremu podataka i trening modela
│   ├── extract_labels.py             parsiranje labela iz imena fajlova
│   ├── feature_extraction.py         Mel karakteristike za KNN/NB
│   ├── feature_extraction_cnn.py     Mel spektrogrami (sekvence) za CNN
│   ├── split.py                      podela na train/val/test po glumcu (KNN/NB)
│   ├── augmentation.py               augmentacija trening skupa (KNN/NB)
│   ├── augmentation_cnn.py           augmentacija trening skupa (CNN)
│   ├── prepare_for_knn.py            skaliranje i enkodiranje labela
│   ├── train_knn.py
│   ├── train_naive_bayes.py
│   ├── train_cnn.py
│   ├── train_wav2vec2.py
│   ├── test_knn.py
│   └── test_naive_bayes.py
├── models/                           sačuvani modeli, scaler-i, confusion matrice
├── visualization/                    EDA vizuelizacije (waveform, mel-spektrogram)
├── emotion_recognition_colab.ipynb   kompletan pipeline za Google Colab
├── requirements.txt
└── README.md
```

## Pokretanje

### Google Colab

1. Otvoriti [colab.research.google.com](https://colab.research.google.com) i učitati `emotion_recognition_colab.ipynb` (File → Upload notebook).
2. Runtime → Change runtime type → GPU (T4).
3. Runtime → Run all.

Sveska sama preuzima repozitorijum i instalira potrebne pakete — nije potrebna dodatna priprema.

### Lokalno

```bash
pip install -r requirements.txt
cd features

python extract_labels.py
python feature_extraction.py
python split.py
python augmentation.py
python prepare_for_knn.py
python train_knn.py
python train_naive_bayes.py

python feature_extraction_cnn.py
python augmentation_cnn.py
python train_cnn.py

python train_wav2vec2.py   # zahteva torch/transformers/accelerate i GPU
```

## Zavisnosti

Sve u `requirements.txt`. `torch`, `transformers` i `accelerate` su potrebni samo za Wav2Vec2 (`train_wav2vec2.py`) i zahtevaju znatno više resursa (preporučuje se GPU).
