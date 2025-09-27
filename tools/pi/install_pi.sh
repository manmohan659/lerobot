#!/usr/bin/env bash
set -euo pipefail

# Idempotent installer for Raspberry Pi (LeKiwi host)
# Usage: bash tools/pi/install_pi.sh

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "[pi/install] Repo root: $REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Installing..."
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-venv python3-pip
fi

if ! command -v tailscale >/dev/null 2>&1; then
  echo "[pi/install] Installing Tailscale..."
  curl -fsSL https://tailscale.com/install.sh | sh
  echo "[pi/install] To bring Tailscale up: sudo tailscale up --ssh"
fi

# Create venv if missing
VENV_DIR="${REPO_ROOT}/.venv-lerobot"
if [ ! -d "$VENV_DIR" ]; then
  echo "[pi/install] Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "[pi/install] Installing lerobot with lekiwi extras..."
python -m pip install -e .[lekiwi]

echo "[pi/install] Ensuring pyzmq installed..."
python -m pip install pyzmq

# Serial permissions (Linux)
if id -nG "$USER" | grep -vqE '\bdialout\b'; then
  echo "[pi/install] Adding $USER to dialout for serial ports (requires re-login)"
  sudo usermod -a -G dialout "$USER" || true
fi

echo "[pi/install] Done. To start host: bash tools/pi/start_pi.sh"


