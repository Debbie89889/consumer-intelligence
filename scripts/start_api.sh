#!/usr/bin/env sh
set -e
# Load the precomputed outputs into the database, then serve the API.
# $PORT is provided by the platform (Render); defaults to 8000 locally.
python scripts/load_db.py
exec uvicorn consumer_intel.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
