#!/usr/bin/env bash
# Container entrypoint: init | api | dashboard | experiments | test | demo | <any command>
set -euo pipefail
cd /app

init() {
  # Build data and train models only if they are missing (results persist in mounted volumes).
  if [ ! -f data/processed/transactions.parquet ]; then
    echo "[init] building dataset (data/raw empty -> simulated data is generated)"
    python -m procure.data.build
  fi
  if [ ! -f models/bundle.joblib ]; then
    echo "[init] tuning + training models (Optuna, ~2-3 min)"
    python -m procure.models.train
    python -m procure.models.evaluate
  fi
  echo "[init] ready"
}

case "${1:-dashboard}" in
  init)        init ;;
  api)         exec python -m procure.api.app ;;
  dashboard)   exec streamlit run dashboard/streamlit_app.py --server.headless true \
                 --server.address 0.0.0.0 --server.port 8501 ;;
  experiments) exec python experiments/run_all.py ;;
  test)        exec python -m pytest ;;
  demo)        init && exec python -m procure.pipeline ;;
  *)           exec "$@" ;;
esac
