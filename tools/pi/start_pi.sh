#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV_DIR="${REPO_ROOT}/.venv-lerobot"
ROBOT_ID="${ROBOT_ID:-my_awesome_kiwi}"

source "$VENV_DIR/bin/activate"

echo "[pi/start] Starting LeKiwi host for $ROBOT_ID"
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id="$ROBOT_ID"


