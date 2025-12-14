import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Параметри
SAMPLE_RATE = 16000
N_MFCC = 13
DATASET_PATH = "dataset"

LABELS = ["yes", "no"]
LABEL_MAP = {"yes": 0, "no": 1}

X = []
y = []

# Завантаження датасету
for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder, filename)

            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=N_MFCC
            )

            mfcc_mean = np.mean(mfcc.T, axis=0)

            X.append(mfcc_mean)
            y.append(LABEL_MAP[label])  # ❗ ТІЛЬКИ INT

# Приведення до NumPy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Розбиття
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Модель
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MFCC,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Навчання
model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Збереження
model.export("yes_no_model")