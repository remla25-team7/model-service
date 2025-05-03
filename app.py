from flask import Flask, request, jsonify
import joblib
import os
import requests
from lib_ml.preprocessing import clean_text

MODEL_URL = os.environ.get('MODEL_URL', 'https://github.com/remla25-team7/model-training/releases/download/0.1.0/sentiment_model.pkl')
MODEL_PATH = 'sentiment_model.pkl'

# Download the model if it's not already present
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded.")




app = Flask(__name__)


# Load model
model, vectorizer = joblib.load("sentiment_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)