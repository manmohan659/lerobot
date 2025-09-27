#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV_DIR="${REPO_ROOT}/.venv-detector"
PORT="${DETECTOR_PORT:-8080}"
MODEL="${YOLO_MODEL:-yolov8n.pt}"

source "$VENV_DIR/bin/activate"
echo "[h100/start] Starting detector on port $PORT with model $MODEL"
YOLO_MODEL="$MODEL" uvicorn examples.lekiwi.services.detector_server:app --host 0.0.0.0 --port "$PORT"


