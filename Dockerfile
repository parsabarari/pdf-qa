FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirror.abrha.net/repository/pypi/simple

COPY app ./app
RUN mkdir -p data/uploads data/chroma

# Railway injects $PORT. Shell form so the variable is expanded; 8000 is a local fallback.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
