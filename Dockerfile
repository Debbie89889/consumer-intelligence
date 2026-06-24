FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 (Postgres client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Deploy image only needs the *serving* deps (api + copilot + app), NOT the
# training stack (ml: lightgbm/xgboost/shap/lifetimes). The models were run
# offline to produce the parquet outputs; the running services just read them.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,copilot,app]"

# App code + the precomputed outputs the services serve from
COPY scripts ./scripts
COPY db ./db
COPY sql ./sql
COPY app ./app
COPY data/processed ./data/processed

EXPOSE 8000

# Default command (overridden per-service in docker-compose / render.yaml)
CMD ["uvicorn", "consumer_intel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
