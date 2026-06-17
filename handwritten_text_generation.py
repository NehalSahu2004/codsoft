import os
import numpy as np
import tensorflow as tf
import kagglehub

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =====================================
# DOWNLOAD DATASET
# =====================================

path = kagglehub.dataset_download("loritacalloway/the-complete-william-shakespeare-plays")

print("Dataset Path:", path)

# =====================================
# FIND TEXT FILE
# =====================================

text_file = None

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".txt"):
            text_file = os.path.join(root, file)
            break

print("Text File:", text_file)

# =====================================
# LOAD TEXT
# =====================================

with open(
        text_file,
        "r",
        encoding="utf-8",
        errors="ignore"
) as f:
    text = f.read().lower()

print("Text Length:", len(text))

# Use subset for faster training
text = text[:500000]

# =====================================
# CHARACTER VOCABULARY
# =====================================

chars = sorted(list(set(text)))

char_to_idx = {
    c: i for i, c in enumerate(chars)
}

idx_to_char = {
    i: c for i, c in enumerate(chars)
}

vocab_size = len(chars)

print("Vocabulary Size:", vocab_size)

# =====================================
# CREATE SEQUENCES
# =====================================

sequence_length = 100

X = []
y = []

for i in range(
        0,
        len(text) - sequence_length,
        3
):
    seq = text[i:i + sequence_length]
    target = text[i + sequence_length]

    X.append(
        [char_to_idx[c] for c in seq]
    )

    y.append(
        char_to_idx[target]
    )

X = np.array(X)
y = np.array(y)

print("Sequences:", len(X))

# Limit samples for laptop training
max_samples = 20000

X = X[:max_samples]
y = y[:max_samples]

# =====================================
# ONE HOT ENCODING
# =====================================

X = to_categorical(
    X,
    num_classes=vocab_size
)

y = to_categorical(
    y,
    num_classes=vocab_size
)

# =====================================
# BUILD LSTM MODEL
# =====================================

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(
            sequence_length,
            vocab_size
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        vocab_size,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# TRAIN MODEL
# =====================================

model.fit(
    X,
    y,
    epochs=20,
    batch_size=64
)

# =====================================
# TEXT GENERATION
# =====================================

def generate_text(
        seed_text,
        length=500):

    generated = seed_text.lower()

    for _ in range(length):

        sequence = generated[
                   -sequence_length:
                   ]

        encoded = [
            char_to_idx.get(c, 0)
            for c in sequence
        ]

        while len(encoded) < sequence_length:
            encoded.insert(0, 0)

        encoded = to_categorical(
            encoded,
            num_classes=vocab_size
        )

        encoded = np.reshape(
            encoded,
            (
                1,
                sequence_length,
                vocab_size
            )
        )

        prediction = model.predict(
            encoded,
            verbose=0
        )

        next_index = np.argmax(
            prediction
        )

        next_char = idx_to_char[
            next_index
        ]

        generated += next_char

    return generated

# =====================================
# GENERATE SAMPLE TEXT
# =====================================

seed = "to be or not to be"

generated_text = generate_text(
    seed,
    1000
)

print("\nGenerated Text:\n")
print(generated_text)

# Save generated text

with open(
        "generated_text.txt",
        "w",
        encoding="utf-8"
) as f:
    f.write(generated_text)

# Save model

model.save(
    "text_generator_model.keras"
)

print("\nModel Saved Successfully")