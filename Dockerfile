FROM python:3.12.9-slim

ENV MODEL_VERSION="v1.2.1"

ENV PYTHONUNBUFFERED=1 \ 
    MODEL_URL=https://github.com/remla25-team7/model-training/releases/download/${MODEL_VERSION}/model.pkl \
    VECTORIZER_URL=https://github.com/remla25-team7/model-training/releases/download/${MODEL_VERSION}/vectorizer.pkl \
    PORT=8080 \
    IN_DOCKER=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ARG MODEL_SERVICE_VERSION

ENV MODEL_SERVICE_VERSION=$MODEL_SERVICE_VERSION

COPY app.py .

EXPOSE 5000

ENTRYPOINT ["python","app.py"]
