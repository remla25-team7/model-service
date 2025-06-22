import os
import requests
import joblib
import tempfile
import pathlib
from functools import wraps
from lib_ml.preprocessing import clean_review

MODEL_URL      = os.environ["MODEL_URL"]
VECTORIZER_URL = os.environ["VECTORIZER_URL"]
CACHE_DIR      = pathlib.Path("/model-cache") 
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _fetch(url: str, dest: pathlib.Path):
    if dest.exists():
        return dest
    headers = {}
    print(f"Downloading {url} → {dest}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest

model_path      = _fetch(MODEL_URL,      CACHE_DIR / "model.pkl")
vectorizer_path = _fetch(VECTORIZER_URL, CACHE_DIR / "vectorizer.pkl")

model      = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

from flask import Flask, request, jsonify
app = Flask(__name__)


SECRET_PATHS = [
    '/run/secrets/model_credentials',  # docker-compose default
    '/app/secrets/model_credentials',  # kubernetes default
]

API_KEY = None

for path_str in SECRET_PATHS:
    path = pathlib.Path(path_str)
    if path.exists():
        print(f"MODEL SERVICE: Successfully loaded API key from {path_str}.")
        with path.open('r') as f:
            API_KEY = f.read().strip()
        break

if not API_KEY:
    print(f"WARNING: Secret file not found in any known location {SECRET_PATHS}.")

# define the decorator to protect endpoints
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # check if security is enabled and if the key matches
        if API_KEY:
            provided_key = request.headers.get('X-API-Key')
            if not provided_key or provided_key != API_KEY:
                return jsonify({"error": "Unauthorized. Invalid or missing API Key."}), 401
        else:
            return jsonify({"error": "Unauthorized. Invalid or missing API Key."}), 401

        
        # if key is valid proceed with the request
        return f(*args, **kwargs)
    return decorated_function


@app.route("/predict", methods=["POST"])
@api_key_required 
def predict():
    data = request.get_json(silent=True)
    review = data.get("review") if data else None
    if not review:
        return jsonify({"error": "Missing 'review' in JSON body"}), 400

    cleaned = clean_review(review)
    X    = vectorizer.transform([cleaned]).toarray()
    pred = int(model.predict(X)[0])
    return jsonify({"sentiment": pred})

if __name__ == "__main__":
    PORT = int(os.getenv("MODEL_SERVICE_PORT", "5000"))
    app.run(host="0.0.0.0", port=PORT)
