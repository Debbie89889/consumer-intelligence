#!/usr/bin/env sh
set -e
# $PORT is provided by the platform (Render); defaults to 8501 locally.
exec streamlit run app/dashboard.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
