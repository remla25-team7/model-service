FROM python:3.9-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY service service

RUN mkdir -p /app/models

EXPOSE 8080
ENV MODEL_DIR=/app/models
ENV PORT=8080

ENTRYPOINT ["python", "service/app.py"]
