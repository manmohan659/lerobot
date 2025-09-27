#!/usr/bin/env bash
set -euo pipefail

echo "[h100/install] Installing runtime deps..."
if ! command -v python3 >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-venv python3-pip git curl
fi

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
  echo "[h100/install] To bring Tailscale up: sudo tailscale up --ssh"
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV_DIR="${REPO_ROOT}/.venv-detector"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install ultralytics fastapi uvicorn[standard] pillow numpy opencv-python-headless

echo "[h100/install] Done. To start: bash tools/h100/start_h100.sh"


