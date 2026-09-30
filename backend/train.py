"""
Train the Fake News model (TF-IDF + Logistic Regression).

Dataset: Kaggle "Fake and Real News Dataset" -> put Fake.csv and True.csv in ./data/
Run:     python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def load_data() -> pd.DataFrame:
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)
    fake["label"] = 0   # 0 = Fake
    true["label"] = 1   # 1 = Real
    df = pd.concat([fake, true], ignore_index=True)

    # Use title + text when both exist
    title = df["title"].fillna("") if "title" in df.columns else ""
    df["content"] = (title + " " + df["text"].fillna("")).str.strip()
    df = df[df["content"].str.len() > 0]
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = load_data()
    print(f"Total samples: {len(df)}")

    print("Cleaning text...")
    df["clean"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2), max_features=100000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved {MODEL_PATH.name} and {VECTORIZER_PATH.name}")


if __name__ == "__main__":
    main()