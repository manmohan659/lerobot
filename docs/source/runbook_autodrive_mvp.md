# LeKiwi MVP Runbook: "Fetch Tissue"

## What This Does
End-to-end demo where you say "fetch tissue" on the Mac, the robot searches, approaches, picks up the tissue, and returns. H100 runs the detector service; Mac runs the agent; Raspberry Pi runs the LeKiwi host and motors.

## Components & Roles
- **Raspberry Pi**: `lekiwi_host` (ZMQ ports 5555/5556) executes wheel/joint commands.
- **Mac**: Agent loop (`examples/lekiwi/autodrive.py`) for intent, detection client, visual servoing, scripted pick, return.
- **H100**: Detector server (`examples/lekiwi/services/detector_server.py`) providing `/health` and `/detect`.

## Network Topology
- Use Tailscale on all three; star topology via Mac.
- Mac ↔ Pi: ZMQ to `LEKIWI_PI_IP` (ports 5555/5556)
- Mac ↔ H100: HTTP to `http://H100_IP:8080`

## One-Time Install
### H100
```bash
bash tools/h100/install_h100.sh
sudo tailscale up --ssh   # once
```

### Raspberry Pi
```bash
bash tools/pi/install_pi.sh
sudo tailscale up --ssh   # once
```

### Mac
```bash
bash tools/mac/install_mac.sh
```

## Start Services (Each Session)
### 1) Start Detector on H100
```bash
H100_IP=$(tailscale ip -4)
DETECTOR_PORT=8080 YOLO_MODEL=yolov8n.pt bash tools/h100/start_h100.sh
```

### 2) Start LeKiwi Host on Raspberry Pi
```bash
PI_IP=$(tailscale ip -4)
ROBOT_ID=my_awesome_kiwi bash tools/pi/start_pi.sh
```

### 3) Connection Check from Mac
```bash
export LEKIWI_PI_IP=$PI_IP
export H100_IP=$H100_IP
DETECTOR_PORT=8080 bash tools/connection_check.sh
```
Expect: Pi reachable, H100 reachable, detector health OK, ports 5555/5556 open.

### 4) Run Autodrive on Mac
```bash
export LEKIWI_PI_IP=$PI_IP
export DETECTOR_URL=http://$H100_IP:8080
python examples/lekiwi/autodrive.py
```
Type: `fetch tissue` when prompted.

## Runtime Flow (Step-by-Step)
1) **Voice/Intent**:
   - Mac reads text (prompt) and parses intent via `examples/lekiwi/agent/llm.py` → `{task:"fetch", object:"tissue"}`.
2) **Observation**:
   - Mac `LeKiwiClient` pulls observation from Pi (front image + state).
3) **Detection (H100)**:
   - Mac downscales frame to ~320×240 and POSTs base64 JPEG to `http://H100_IP:8080/detect` with labels `["tissue"]`.
   - H100 returns detections `[ {label, score, bbox{x,y,w,h}} ]`.
4) **Visual Servoing (Mac)**:
   - `navigator.compute_action(detections)` computes `{x.vel,y.vel,theta.vel}`:
     - Center the target (y, theta from bbox center error)
     - Move forward until bbox area reaches threshold (slow near target)
5) **Action Send (Mac → Pi)**:
   - Mac sends the action dict to Pi; Pi host writes wheel velocities.
6) **Transition to PICK**:
   - When bbox area > threshold, FSM enters `PICK` and runs `picker.run_pick()`:
     - Open gripper → descend → close → lift
7) **RETURN**:
   - Simple returner uses stored home heading; drives back for a few seconds, then stops.
8) **DONE**:
   - Robot stops; loop exits.

## Messages & Formats
- **Detector request**:
```json
{
  "image_jpeg_base64": "...",
  "labels": ["tissue"]
}
```
- **Detector response**:
```json
{
  "detections": [
    {"label": "tissue", "score": 0.82,
     "bbox": {"x": 320, "y": 240, "w": 180, "h": 120}}
  ],
  "latency_ms": 45.2
}
```
- **Robot action from Mac**:
```json
{"x.vel": 0.2, "y.vel": 0.0, "theta.vel": 10.0}
```

## Safety & Latency Notes
- Approach speeds are limited: `v_max_near = 0.08 m/s` to avoid overruns.
- Detector runs at ~10 Hz; agent loop ~15–30 Hz.
- If detector returns no target, robot scans in place; forward velocity = 0.
- Watchdog on Pi will stop base if no commands for >500 ms.

## What You Should See
- **Mac console**: intent, state transitions, loop logs; Rerun telemetry.
- **H100 console**: detector requests and health pings.
- **Pi console**: host running; no-command warnings when idle.
- **Robot**: search → center/approach → pick → return.

## Troubleshooting
- Detector health fails: ensure H100 port 8080 open; Tailscale up; server running.
- ZMQ timeout on Mac: confirm `LEKIWI_PI_IP` correct; Pi host running.
- No motion: verify commands printed, and Pi host logs; check motor power/IDs.
- Detector misses tissue: improve lighting/placement; adjust label to a class YOLO knows (e.g., `box`, `napkin`) or switch model.

## Tear Down
- Ctrl+C on Mac and H100 terminals.
- Pi host will stop on Ctrl+C.

## Next Improvements (Optional)
- Pi reflex guard (ROI obstacle stop + TTL/seq clamp)
- Learned pickup policy trained on HF SO100 + your demos
- Return via ArUco/person detection
