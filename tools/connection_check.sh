#!/usr/bin/env bash
set -euo pipefail

PI_IP="${LEKIWI_PI_IP:?Set LEKIWI_PI_IP}"
H100_IP="${H100_IP:?Set H100_IP}"
DETECTOR_PORT="${DETECTOR_PORT:-8080}"

echo "[check] Tailscale (if installed)"
if command -v tailscale >/dev/null 2>&1; then
  tailscale status || true
else
  echo "tailscale not installed on this machine"
fi

echo "[check] Ping Pi: $PI_IP"
ping -c1 "$PI_IP" >/dev/null && echo "Pi reachable" || echo "Pi unreachable"

echo "[check] Ping H100: $H100_IP"
ping -c1 "$H100_IP" >/dev/null && echo "H100 reachable" || echo "H100 unreachable"

echo "[check] Detector health: http://$H100_IP:$DETECTOR_PORT/health"
curl -sfS "http://$H100_IP:$DETECTOR_PORT/health" && echo || echo "Detector health failed"

echo "[check] Pi ZMQ ports"
if command -v nc >/dev/null 2>&1; then
  nc -zv "$PI_IP" 5555 && echo "port 5555 open" || echo "port 5555 closed"
  nc -zv "$PI_IP" 5556 && echo "port 5556 open" || echo "port 5556 closed"
else
  echo "nc not installed; skipping port check"
fi

echo "[check] Done"


