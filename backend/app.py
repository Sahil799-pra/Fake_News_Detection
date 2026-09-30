"""
Flask REST API for Fake News Detection.
Run: python app.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from preprocess import clean_text

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

MIN_WORDS = 10
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def ensure_model_ready():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        print("Model files not found. Training model...")
        import train
        train.main()

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files still missing after training.")

    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


model, vectorizer = ensure_model_ready()

# ---- Optional MongoDB (prediction history). Enable by setting MONGO_URI ----
history_col = None
if os.getenv("MONGO_URI"):
    try:
        from pymongo import MongoClient
        history_col = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=3000)["fake_news_db"]["predictions"]
        print("MongoDB connected: saving prediction history")
    except Exception as e:
        print("MongoDB not available, continuing without history:", e)


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text.split()) < MIN_WORDS:
        return jsonify({"error": f"Please enter at least {MIN_WORDS} words for a reliable prediction"}), 400

    vec = vectorizer.transform([clean_text(text)])
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    model_prediction = "Real" if label == 1 else "Fake"
    confidence = round(float(proba[label]) * 100, 2)

    prediction = model_prediction
    if GEMINI_API_KEY:
        ai_prediction = classify_with_gemini(text)
        if ai_prediction:
            prediction = ai_prediction

    ai_explanation = ""
    if GEMINI_API_KEY:
        ai_explanation = get_gemini_explanation(text, prediction, confidence)

    if history_col is not None:
        try:
            history_col.insert_one({
                "text": text,
                "prediction": prediction,
                "confidence": confidence,
                "created_at": datetime.now(timezone.utc),
            })
        except Exception as e:
            print("History save failed:", e)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "ai_explanation": ai_explanation,
    })


def gemini_call(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return ""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 200,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib_request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body)
            candidates = result.get("candidates") or []
            if not candidates:
                return ""
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    except (urllib_error.HTTPError, urllib_error.URLError, ValueError, KeyError, TypeError):
        return ""


def classify_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY:
        return ""

    prompt = (
        "You are a fake-news analyst. Analyze the article and decide if it is Fake or Real. "
        "Respond with exactly one word: Fake or Real. Do not add any extra text.\n\nArticle:\n"
        + text
    )
    response = gemini_call(prompt)
    if not response:
        return ""

    cleaned = response.strip().replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.split()[0].upper()
    if cleaned in {"FAKE", "REAL"}:
        return cleaned.title()
    if "FAKE" in cleaned.upper():
        return "Fake"
    if "REAL" in cleaned.upper():
        return "Real"
    return ""


def get_gemini_explanation(text: str, prediction: str, confidence: float) -> str:
    if not GEMINI_API_KEY:
        return "AI explanation unavailable: Gemini API key not set."

    prompt = (
        "You are helping review a news article. "
        f"The model predicted it as {prediction} with {confidence}% confidence. "
        "Give a brief, balanced explanation in 2-4 sentences. "
        "Focus on suspicious signs like sensational language, missing evidence, lack of sources, or strong claims. "
        "Do not claim certainty beyond the text.\n\nArticle:\n"
        + text
    )
    result = gemini_call(prompt)
    return result or "AI explanation unavailable: Gemini returned no response."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)