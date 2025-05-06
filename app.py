import os
import tarfile
import tempfile
from pathlib import Path

import joblib
import requests
from flask import Flask, request, jsonify
from lib_ml.preprocessing import tokenize_review

# CONFIGURATION (via env vars)
MODEL_URL = os.getenv("MODEL_URL")  # e.g. "https://github.com/.../models.tar.gz"
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
PORT = int(os.getenv("PORT", 8080))

if not MODEL_URL:
    raise RuntimeError("MODEL_URL environment variable is not set")

def fetch_and_extract_model(url: str, dest: Path):
    """Download a .tar.gz from `url` and extract into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # write to temp file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            for chunk in r.iter_content(1024):
                tmp.write(chunk)
            tmp_path = tmp.name

    # extract
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(path=dest.parent)
    os.remove(tmp_path)

# Ensure the model artifacts are present
if not (MODEL_DIR / "vectorizer.pkl").exists() or not (MODEL_DIR / "classifier.joblib").exists():
    fetch_and_extract_model(MODEL_URL, MODEL_DIR)

# Load the vectorizer and classifier
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
classifier = joblib.load(MODEL_DIR / "classifier.joblib")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("review_text", "")
    X = vectorizer.transform([text])
    pred = classifier.predict(X)[0]
    return jsonify({"prediction": pred})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)