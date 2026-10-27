FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-tha \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY reindex.py check.py ./

RUN useradd --create-home --uid 10001 rise \
    && mkdir -p /app/data/ingestion /app/data/private /app/logs \
    && chown -R rise:rise /app/data /app/logs

USER rise
EXPOSE 8077

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077", "--workers", "1", "--no-proxy-headers"]

