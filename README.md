# 🔍 Sentiment Analysis Service

Production-ready Flask API that delivers sentiment predictions using a pre-trained scikit-learn model and integrated text vectorization via `lib-ml`.

## 📌 Features

- **Text Preprocessing**: Uses `lib-ml` vectorizer for consistent input transformation
- **Model Inference**: Serves a pre-trained classifier with versioned model loading
- **REST API Endpoints**:
  - `/predict` – Run sentiment prediction on input text
- **Health Check**: Minimal footprint for production readiness
- **Environment-Driven**: Configure model sources and port via environment variables

## 🚀 Deployment

### Docker (Recommended)

#### 1. Build the Docker image

```bash
docker build -t model-service .
```

#### 2. Run the Docker Image

```bash
docker run -p 5000:5000 \
  -e MODEL_URL=https://github.com/remla25-team7/model-training/releases/download/v0.3.0/model.pkl \
  -e VECTORIZER_URL=https://github.com/remla25-team7/model-training/releases/download/v0.3.0/vectorizer.pkl \
  -e PORT=5000 \
  model-service
```

### 3. Troubleshooting

On macOS, port **5000** may be reserved by system services such as AirPlay or Control Center. If you get “port already in use” errors:

- Check which process uses port 5000:

  ```bash
  lsof -i :5000
  ```

  Disable AirPlay Receiver in System Settings > AirPlay & Handoff to free the port.
  Or run the container on a different port, e.g., 8080:
