FROM python:3.12.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY app.py .

EXPOSE 8080        

ENTRYPOINT ["python","app.py"]
