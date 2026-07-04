import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ----------------------------
# Load Dataset
# ----------------------------

DATA_PATH = "data/processed/processed_emotions.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.shape)

# ----------------------------
# Encode Labels
# ----------------------------

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["emotion"])

NUM_LABELS = len(label_encoder.classes_)

print("Classes:")
print(label_encoder.classes_)

# ----------------------------
# Train/Test Split
# ----------------------------

# ----------------------------
# Use Smaller Dataset
# ----------------------------

df = df.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# ----------------------------
# Load Tokenizer
# ----------------------------

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------
# Tokenization
# ----------------------------

def tokenize(batch):
    return tokenizer(
        batch["processed_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)

print("Dataset Ready!")
# ----------------------------
# Load BERT Model
# ----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# ----------------------------
# Metrics
# ----------------------------

import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    return metric.compute(
        predictions=predictions,
        references=labels
    )


# ----------------------------
# Training Arguments
# ----------------------------

training_args = TrainingArguments(
    output_dir="saved_models/bert_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# ----------------------------
# Trainer
# ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train
# ----------------------------

print("\nStarting BERT Training...\n")

trainer.train()

# ----------------------------
# Evaluate
# ----------------------------

results = trainer.evaluate()

print("\nEvaluation Results")

print(results)

# ----------------------------
# Save Model
# ----------------------------

SAVE_DIR = "saved_models/bert"

trainer.save_model(SAVE_DIR)

tokenizer.save_pretrained(SAVE_DIR)

import pickle

with open("saved_models/bert_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n====================================")
print("BERT MODEL SAVED SUCCESSFULLY")
print("====================================")

print(f"Saved to : {SAVE_DIR}")