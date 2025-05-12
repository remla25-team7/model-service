import os, requests, joblib, tempfile, pathlib

MODEL_URL      = os.environ["MODEL_URL"]
VECTORIZER_URL = os.environ["VECTORIZER_URL"]
CACHE_DIR      = pathlib.Path("/tmp/artefacts") 
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _fetch(url: str, dest: pathlib.Path):
    if dest.exists():
        return dest
    headers = {}
    if "GITHUB_TOKEN" in os.environ:       
        headers["Authorization"] = f'token {os.environ["GITHUB_TOKEN"]}'
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

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    X = vectorizer.transform([text]).toarray()
    pred = int(model.predict(X)[0])
    return jsonify({"review": pred})

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=PORT)
