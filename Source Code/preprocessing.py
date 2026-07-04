import os
import re
import pandas as pd

# -------------------------------
# Paths
# -------------------------------

RAW_DATA = "data/raw/emotions.csv"
PROCESSED_FOLDER = "data/processed"
PROCESSED_DATA = "data/processed/processed_emotions.csv"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# -------------------------------
# Clean Text
# -------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# Emotion Mapping
# -------------------------------

def map_emotion(row):

    if row["confusion"] == 1:
        return "Confused"

    elif row["curiosity"] == 1:
        return "Curious"

    elif (
        row["joy"] == 1
        or row["optimism"] == 1
        or row["approval"] == 1
        or row["admiration"] == 1
        or row["pride"] == 1
    ):
        return "Confident"

    elif (
        row["anger"] == 1
        or row["annoyance"] == 1
        or row["disappointment"] == 1
        or row["disapproval"] == 1
        or row["frustration"] == 1 if "frustration" in row else False
        or row["sadness"] == 1
    ):
        return "Frustrated"

    elif (
        row["neutral"] == 1
        or row["disinterest"] == 1 if "disinterest" in row else False
    ):
        return "Bored"

    else:
        return None


# -------------------------------
# Preprocess Dataset
# -------------------------------

def preprocess_dataset():

    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA)

    print(f"Rows Loaded : {len(df)}")

    df["processed_text"] = df["text"].apply(clean_text)

    df["emotion"] = df.apply(map_emotion, axis=1)

    df = df.dropna(subset=["emotion"])

    df = df[["processed_text", "emotion"]]

    df.to_csv(PROCESSED_DATA, index=False)

    print("Dataset Saved Successfully!")

    print(PROCESSED_DATA)

    print(df.head())

    print()

    print(df["emotion"].value_counts())


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    preprocess_dataset()