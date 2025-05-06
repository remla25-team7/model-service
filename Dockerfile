FROM python:3.9-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=base /usr/local/lib/python3.9/site-packages \
                    /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin/ /usr/local/bin/


COPY app.py .

EXPOSE 8080
CMD ["gunicorn", "app:app", "--workers", "2", "--bind", "0.0.0.0:8080"]
