import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -----------------------
# Paths
# -----------------------

DATA_PATH = "data/processed/processed_emotions.csv"

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# -----------------------
# Load dataset
# -----------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

texts = df["processed_text"].astype(str)

labels = df["emotion"]

print(df.head())

# -----------------------
# Encode labels
# -----------------------

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

y = to_categorical(y)

# -----------------------
# Tokenizer
# -----------------------

VOCAB_SIZE = 10000
MAX_LENGTH = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    X,
    maxlen=MAX_LENGTH,
    padding="post"
)

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Build Model
# -----------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        input_length=MAX_LENGTH
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(Dropout(0.4))

model.add(Dense(64, activation="relu"))

model.add(Dense(y.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Train
# -----------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64
)

# -----------------------
# Evaluate
# -----------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nAccuracy : {accuracy*100:.2f}%")

# -----------------------
# Save
# -----------------------

model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)

print("\nModel Saved Successfully!")

print(MODEL_PATH)
print(TOKENIZER_PATH)
print(LABEL_ENCODER_PATH)