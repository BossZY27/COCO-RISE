FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv/rise
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tha tesseract-ocr-eng curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini .
COPY data ./data
COPY docs ./docs
COPY funds-data ./funds-data
RUN useradd --create-home --uid 10001 rise && mkdir -p /srv/rise/data/private /srv/rise/data/chroma_db /srv/rise/logs && chown -R rise:rise /srv/rise
USER rise
EXPOSE 8077
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077", "--workers", "1"]
