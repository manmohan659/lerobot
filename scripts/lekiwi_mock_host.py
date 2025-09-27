#!/usr/bin/env python3

"""
LeKiwi Mock Host - For testing network connection without real motors
Run this on Pi 5 to test ZMQ communication before setting up real motors
"""

import json
import time
import base64
import numpy as np
import cv2
import zmq

def create_mock_observation():
    """Create a mock observation similar to real LeKiwi"""

    # Mock robot state (9 values for LeKiwi)
    mock_state = {
        "arm_shoulder_pan.pos": 0.1 * np.sin(time.time()),
        "arm_shoulder_lift.pos": 0.1 * np.cos(time.time()),
        "arm_elbow_flex.pos": 0.05 * np.sin(time.time() * 1.5),
        "arm_wrist_flex.pos": 0.05 * np.cos(time.time() * 1.5),
        "arm_wrist_roll.pos": 0.02 * np.sin(time.time() * 2),
        "arm_gripper.pos": 0.5,
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }

    # Create mock camera image (640x480)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some visual pattern
    cv2.circle(img, (320, 240), 50, (0, 255, 0), -1)
    cv2.putText(img, f"Mock Camera {time.time():.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Encode image as base64 (like real LeKiwi)
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Combine state and image
    observation = {**mock_state, "front": img_b64}

    return observation

def main():
    print("🤖 LeKiwi Mock Host - Testing Network Connection")
    print("=" * 60)
    print("⚠️  This is for testing only - no real motors!")

    # ZMQ setup (same ports as real LeKiwi)
    context = zmq.Context()

    # Command receiver (receives actions from client)
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.bind("tcp://*:5555")
    cmd_socket.setsockopt(zmq.CONFLATE, 1)

    # Observation sender (sends camera + state to client)
    obs_socket = context.socket(zmq.PUSH)
    obs_socket.bind("tcp://*:5556")
    obs_socket.setsockopt(zmq.CONFLATE, 1)

    print("✅ ZMQ sockets bound:")
    print("   Commands: tcp://*:5555")
    print("   Observations: tcp://*:5556")
    print("\n🚀 Starting mock robot loop...")
    print("Ctrl+C to stop")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # Check for incoming commands (non-blocking)
            try:
                cmd_data = cmd_socket.recv_string(zmq.NOBLOCK)
                cmd = json.loads(cmd_data)
                print(f"📨 Received action: {cmd}")
            except zmq.Again:
                pass  # No command received
            except json.JSONDecodeError as e:
                print(f"❌ Invalid command JSON: {e}")

            # Create and send mock observation
            observation = create_mock_observation()
            obs_json = json.dumps(observation)
            obs_socket.send_string(obs_json)

            frame_count += 1

            # Status every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Frame {frame_count:4d} | FPS: {fps:5.1f} | Mock robot running...")

            # Run at ~30 FPS
            sleep_time = 1/30 - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Stopping mock host...")

    finally:
        cmd_socket.close()
        obs_socket.close()
        context.term()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n🏁 Mock host stopped:")
        print(f"   Total frames: {frame_count}")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("✅ Network test completed!")

if __name__ == "__main__":
    main()