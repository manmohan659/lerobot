#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV_DIR="${REPO_ROOT}/.venv-lerobot-mac"
PI_IP="${LEKIWI_PI_IP:?Set LEKIWI_PI_IP to Pi Tailscale IP}"
DETECTOR_URL="${DETECTOR_URL:?Set DETECTOR_URL to H100 detector URL}"

source "$VENV_DIR/bin/activate"
export LEKIWI_PI_IP="$PI_IP"
export DETECTOR_URL="$DETECTOR_URL"

python examples/lekiwi/autodrive.py


