import os
import requests
import joblib
import tempfile
import pathlib
from functools import wraps

MODEL_URL      = os.environ["MODEL_URL"]
VECTORIZER_URL = os.environ["VECTORIZER_URL"]
CACHE_DIR      = pathlib.Path("/tmp/artefacts") 
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


SECRET_FILE_PATH = '/run/secrets/model_credentials'
API_KEY = None

try:
    with open(SECRET_FILE_PATH, 'r') as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    print(f"WARNING: Secret file not found at {SECRET_FILE_PATH}. API key security is disabled.")

# define the decorator to protect endpoints
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # check if security is enabled and if the key matches
        if API_KEY:
            provided_key = request.headers.get('X-API-Key')
            print(f"MODEL SERVICE: My key is '{API_KEY}'")
            print(f"MODEL SERVICE: Received key is '{provided_key}'")
            if not provided_key or provided_key != API_KEY:
                return jsonify({"error": "Unauthorized. Invalid or missing API Key."}), 401
        
        # if key is valid or security is disabled, proceed with the request
        return f(*args, **kwargs)
    return decorated_function


@app.route("/predict", methods=["POST"])
@api_key_required 
def predict():
    data = request.get_json(silent=True)
    review = data.get("review") if data else None
    if not review:
        return jsonify({"error": "Missing 'review' in JSON body"}), 400

    X    = vectorizer.transform([review]).toarray()
    pred = int(model.predict(X)[0])
    return jsonify({"sentiment": pred})

if __name__ == "__main__":
    PORT = int(os.getenv("MODEL_SERVICE_PORT", "5000"))
    app.run(host="0.0.0.0", port=PORT)
