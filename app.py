import os
import tempfile
from pathlib import Path

import joblib
import requests
from flask import Flask, request, jsonify


VECTORIZER_URL = os.getenv(
    "VECTORIZER_URL",
    "https://github.com/remla25-team7/model-training/releases/download/0.1.0-test/vectorizer.pkl"
)
MODEL_URL      = os.getenv(
    "MODEL_URL",
    "https://github.com/remla25-team7/model-training/releases/download/0.1.0-test/model.pkl"
)
MODEL_DIR      = Path(os.getenv("MODEL_DIR", "/models"))
PORT           = int(os.getenv("PORT", 8080))
TIMEOUT        = int(os.getenv("DL_TIMEOUT", 15))

# sanity check
if not VECTORIZER_URL or not MODEL_URL:
    raise RuntimeError("Both VECTORIZER_URL and MODEL_URL env vars must be set")


def fetch_file(url: str, dest: Path) -> None:
    """Download `url` into `dest` unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"[model-service] Downloading {url} → {dest}")
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
    # move into place
    dest.write_bytes(Path(tmp_path).read_bytes())
    os.remove(tmp_path)


fetch_file(VECTORIZER_URL, MODEL_DIR / "vectorizer.pkl")
fetch_file(MODEL_URL,      MODEL_DIR / "model.pkl")

from lib_ml.preprocessing import tokenize_review

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
classifier = joblib.load(MODEL_DIR / "model.pkl")


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("review_text", "")
    X = vectorizer.transform([text])
    pred = int(classifier.predict(X)[0])
    return jsonify({"review": pred})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
