
# Model Service (Sentiment Predictor)

This service powers the sentiment prediction logic for the application. It exposes a simple REST API that receives review texts, transforms them using a pre-trained vectorizer, and returns a binary sentiment classification.

---

## Features

- `/predict`: Accepts a review string and returns the predicted sentiment (`0` = negative, `1` = positive)
- `/version`: Returns the model-service version
- API Key protected (`X-API-Key`)
- Swagger (OpenAPI) documentation at `/apidocs/`

---

## API Docs (Swagger UI)

Start the service and go to:

```
http://localhost:5000/apidocs/
```

Here you can view and test the available endpoints using the interactive Swagger UI.

---

## API Key Security

All POST requests to `/predict` must include a valid `X-API-Key` header. The service loads this key from a secret file, depending on the environment:

- **Docker Compose:** `/run/secrets/model_credentials`
- **Kubernetes:** `/app/secrets/model_credentials`

If the API key is missing or incorrect, the service will return a `401 Unauthorized` error.

---

## Configuration

This service uses environment variables to locate and cache the model files:

```env
MODEL_URL=https://.../model.pkl
VECTORIZER_URL=https://.../vectorizer.pkl
MODEL_SERVICE_PORT=5000
IN_DOCKER=0
MODEL_SERVICE_VERSION=v1.2.3
```

- `MODEL_URL`, `VECTORIZER_URL`: URLs to fetch the model artifacts
- `MODEL_SERVICE_PORT`: The port to run the Flask server on
- `IN_DOCKER`: If `1`, use `/model-cache` inside the container; otherwise, use `./model-cache` locally
- `MODEL_SERVICE_VERSION`: Version string for the `/version` endpoint

---

## Running Locally (Standalone)

You can run the model-service without Docker for testing:

1. Ensure Python 3.12+ is installed.
2. Create and activate a virtual environment.
3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
4. Start the service:
  ```bash
  python app.py
  ```

Now access:

```
http://localhost:5000/apidocs/
```

---


