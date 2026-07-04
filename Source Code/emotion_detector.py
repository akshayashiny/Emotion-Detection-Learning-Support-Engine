import pickle
import numpy as np
import torch
import streamlit as st

from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Paths
# -----------------------

MODEL_PATH = "saved_models/bilstm_model.keras"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"
BERT_MODEL_PATH = "saved_models/bert_emotion_model"

MAX_LENGTH = 50

# -----------------------
# Load Label Encoder
# -----------------------

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------
# Cache BiLSTM Resources
# -----------------------

@st.cache_resource
def load_bilstm():

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

# -----------------------
# Cache BERT Resources
# -----------------------

@st.cache_resource
def load_bert():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_PATH
    )

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device

# -----------------------
# BiLSTM Prediction
# -----------------------

def detect_emotion(text):

    model, tokenizer = load_bilstm()

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    emotion = label_encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    confidence = float(np.max(prediction))

    return emotion, confidence

# -----------------------
# BERT Prediction
# -----------------------

def detect_emotion_bert(text):

    model, tokenizer, device = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    emotion = label_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = float(
        probabilities[0][prediction]
    )

    return emotion, confidence