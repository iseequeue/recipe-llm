#!/usr/bin/env bash
# Open http://localhost:5000 (use ssh -L 5000:localhost:5000 for a remote box)
cd "$(dirname "$0")/.."
mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}" --host 0.0.0.0 --port "${PORT:-5000}"
