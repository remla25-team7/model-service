import os
import tarfile
import tempfile
from pathlib import Path

import joblib
import requests
from flask import Flask, request, jsonify


MODEL_URL  = os.getenv("MODEL_URL")         
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "/models"))
PORT       = int(os.getenv("PORT", 8080))  
TIMEOUT    = int(os.getenv("DL_TIMEOUT", 15))

if not MODEL_URL:
    raise RuntimeError("MODEL_URL environment variable must be set")

def fetch_and_extract(url: str, dest: Path) -> None:
    """Download a .tar.gz from `url` into `dest` unless it is already there."""
    dest.mkdir(parents=True, exist_ok=True)
    vectorizer_file = dest / "vectorizer.pkl"
    classifier_file = dest / "model.pkl"
    if vectorizer_file.exists() and classifier_file.exists():
        return                         
    print(f"[model‑service] Downloading model from {url}")
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(path=dest)
    os.remove(tmp_path)


fetch_and_extract(MODEL_URL, MODEL_DIR)

from lib_ml.preprocessing import tokenize_review  

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
classifier = joblib.load(MODEL_DIR / "model.pkl")


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """Return 1 for positive, 0 for negative sentiment."""
    body = request.get_json(force=True)
    text = body.get("review_text", "")
    X = vectorizer.transform([text])
    y = classifier.predict(X)[0]
    return jsonify({"review": int(y)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
