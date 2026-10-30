
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir mlflow psycopg2-binary

EXPOSE 5000

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"postgresql://mlflow:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow\" --artifacts-destination /mlartifacts"]
