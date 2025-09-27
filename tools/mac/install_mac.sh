#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on Mac. Install Xcode CLT / Homebrew python3 first."
  exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
  echo "Install Tailscale from https://tailscale.com/download/macos or 'brew install tailscale'"
fi

VENV_DIR="${REPO_ROOT}/.venv-lerobot-mac"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .[lekiwi] requests pydantic opencv-python-headless

echo "[mac/install] Done. To start: bash tools/mac/start_mac.sh"


