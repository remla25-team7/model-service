FROM python:3.12.9-slim

ENV PYTHONUNBUFFERED=1 \ 
    MODEL_URL=https://github.com/remla25-team7/model-training/releases/download/v0.3.0/model.pkl \
    VECTORIZER_URL=https://github.com/remla25-team7/model-training/releases/download/v0.3.0/vectorizer.pkl \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY app.py .

EXPOSE 5000

ENTRYPOINT ["python","app.py"]
