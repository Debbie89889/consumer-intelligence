FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 (Postgres client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the package with API + ML + Copilot extras
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,ml,copilot]"

# App code (scripts, data are mounted/loaded at runtime)
COPY scripts ./scripts
COPY db ./db
COPY sql ./sql

EXPOSE 8000

CMD ["uvicorn", "consumer_intel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
