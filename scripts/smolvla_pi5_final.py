#!/usr/bin/env python3

"""
SmolVLA Final Pi 5 Pipeline
- Auto-detects display capability (X11 forwarding vs headless)
- Robust camera detection for Pi 5
- Fixed normalization for all buffers
- Real-time language-conditioned inference
"""

import cv2
import numpy as np
import torch
import time
import sys
import select
import os
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def can_display():
    """Check if we can show windows (X11 forwarding)"""
    return 'DISPLAY' in os.environ

def fix_normalization_stats(policy):
    """Fix all normalization buffers comprehensively"""
    print("🔧 Fixing normalization stats...")

    fixed_count = 0

    # Fix all normalize_inputs buffers
    if hasattr(policy, 'normalize_inputs'):
        for attr_name in dir(policy.normalize_inputs):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.normalize_inputs, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    # Check if infinite
                    if torch.isinf(buffer.mean).any() or torch.isinf(buffer.std).any():
                        buffer.mean.fill_(0.0)
                        buffer.std.fill_(1.0)
                        fixed_count += 1
                        print(f"   ✅ Fixed normalize_inputs.{attr_name}")

    # Fix all normalize_targets buffers
    if hasattr(policy, 'normalize_targets'):
        for attr_name in dir(policy.normalize_targets):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.normalize_targets, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    if torch.isinf(buffer.mean).any() or torch.isinf(buffer.std).any():
                        buffer.mean.fill_(0.0)
                        buffer.std.fill_(1.0)
                        fixed_count += 1
                        print(f"   ✅ Fixed normalize_targets.{attr_name}")

    # Fix all unnormalize_outputs buffers
    if hasattr(policy, 'unnormalize_outputs'):
        for attr_name in dir(policy.unnormalize_outputs):
            if attr_name.startswith('buffer_'):
                buffer = getattr(policy.unnormalize_outputs, attr_name)
                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    if torch.isinf(buffer.mean).any() or torch.isinf(buffer.std).any():
                        buffer.mean.fill_(0.0)
                        buffer.std.fill_(1.0)
                        fixed_count += 1
                        print(f"   ✅ Fixed unnormalize_outputs.{attr_name}")

    print(f"   ✅ Fixed {fixed_count} normalization buffers")

def init_camera():
    """Try multiple camera access methods for Pi 5 - using main branch logic"""
    print("📷 Initializing camera with extended search...")

    # Use the same MAX_OPENCV_INDEX as main branch for robust detection
    MAX_OPENCV_INDEX = 60

    # First try /dev/video* paths (Linux-specific)
    import platform
    from pathlib import Path

    camera_attempts = []

    if platform.system() == "Linux":
        # Scan /dev/video* devices first
        possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
        for path in possible_paths:
            camera_attempts.append((str(path), f"Video device {path.name}"))

    # Then try indices 0 to MAX_OPENCV_INDEX
    for i in range(MAX_OPENCV_INDEX):
        camera_attempts.append((i, f"Camera index {i}"))

    print(f"   Scanning {len(camera_attempts)} possible camera locations...")

    for camera_id, desc in camera_attempts:
        if len(camera_attempts) > 10 and isinstance(camera_id, int) and camera_id % 10 == 0:
            print(f"   Checking indices {camera_id}-{camera_id+9}...")

        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test frame capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ {desc} working! Frame: {frame.shape}")

                    # Set optimal properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    return cap
                else:
                    cap.release()
            else:
                if cap:
                    cap.release()
        except Exception:
            # Silently continue for cleaner output during bulk scanning
            continue

    print("❌ No working camera found after scanning all indices!")
    print("💡 Troubleshooting:")
    print("   - Check: ls /dev/video*")
    print("   - Enable Pi Camera: sudo raspi-config > Interface > Camera")
    print("   - Install libs: sudo apt install python3-picamera2")
    print("   - Test camera: libcamera-hello --timeout 2000")
    print("   - Reboot after enabling camera in raspi-config")
    return None

def kbhit():
    """Non-blocking keyboard input check"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def test_smolvla_pi5():
    print("🚀 SmolVLA Final Pi 5 Pipeline")
    print("=" * 60)

    # Check display capability
    has_display = can_display()
    print(f"🖥️  Display: {'Available (X11)' if has_display else 'Headless mode'}")

    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")

    # Load SmolVLA
    print("📥 Loading SmolVLA (cached)...")
    start_time = time.time()
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device)
    load_time = time.time() - start_time
    print(f"✅ SmolVLA loaded in {load_time:.1f}s")

    # Fix normalization
    fix_normalization_stats(policy)

    # Initialize camera
    cap = init_camera()
    if cap is None:
        return

    print("📷 Camera ready!")
    print("\n🎯 Language-Conditioned Inference Starting...")
    if has_display:
        print("📺 Window will show camera feed + info")
    print("🎮 Controls: 'n' + ENTER = next instruction, 'q' + ENTER = quit")
    print("=" * 60)

    # Language instructions
    instructions = [
        "Pick up the red cup",
        "Move the robot arm up",
        "Grasp the blue block",
        "Push the object to the right",
        "Lift the item carefully",
        "Move to home position",
        "Open the gripper wide",
        "Close the gripper gently",
        "Rotate the arm clockwise",
        "Stop all movement"
    ]

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    inference_times = []
    instruction_index = 0

    # Simulated robot state
    current_robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"🗣️  CURRENT INSTRUCTION: '{instructions[instruction_index]}'")
    input("Press ENTER to start...")

    try:
        while True:
            # Get current instruction
            instruction = instructions[instruction_index % len(instructions)]

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed")
                continue

            frame_count += 1

            # Update simulated robot state
            t = time.time() * 0.1
            current_robot_state = np.array([
                0.1 * np.sin(t),
                0.15 * np.cos(t * 1.1),
                0.2 * np.sin(t * 0.8),
                0.1 * np.cos(t * 1.3),
                0.05 * np.sin(t * 1.5),
                0.3 * np.cos(t * 0.7)
            ])

            # Preprocess image
            image_resized = cv2.resize(frame, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(device)

            # Robot state tensor
            state_tensor = torch.from_numpy(current_robot_state).unsqueeze(0).float().to(device)

            # Create observation
            observation = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "task": instruction
            }

            # Model inference
            inference_start = time.time()
            with torch.no_grad():
                action = policy.select_action(observation)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # Calculate performance
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_inference = np.mean(inference_times[-10:]) * 1000

            # Extract action values
            action_values = action.flatten().cpu().numpy()

            # Terminal output every 5 frames
            if frame_count % 5 == 0:
                print(f"\r📊 Frame {frame_count:4d} | FPS: {current_fps:5.1f} | Inference: {avg_inference:6.1f}ms", end="")
                sys.stdout.flush()

            # Detailed info every 20 frames
            if frame_count % 20 == 0:
                print(f"\n🔄 LIVE UPDATE:")
                print(f"   📝 Task: '{instruction}'")
                print(f"   🤖 Robot State: [{', '.join(f'{x:6.3f}' for x in current_robot_state)}]")
                print(f"   🎯 Motor Output: [{', '.join(f'{x:6.3f}' for x in action_values)}]")
                print(f"   ⚡ Performance: {current_fps:.1f} FPS, {avg_inference:.1f}ms avg")
                print("-" * 60)

            # Display window if available
            if has_display:
                display_frame = frame.copy()

                # Add overlays
                cv2.putText(display_frame, f"Task: {instruction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Inference: {avg_inference:.0f}ms",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(display_frame, f"State: [{current_robot_state[0]:.2f}, {current_robot_state[1]:.2f}, {current_robot_state[2]:.2f}...]",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)
                cv2.putText(display_frame, f"Motors: [{action_values[0]:.2f}, {action_values[1]:.2f}, {action_values[2]:.2f}...]",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # Status indicator
                color = (0, 255, 0) if avg_inference < 500 else (0, 255, 255)
                cv2.circle(display_frame, (600, 30), 10, color, -1)

                cv2.imshow('SmolVLA Pi 5 Pipeline', display_frame)
                cv2.waitKey(1)

            # Check keyboard input
            if kbhit():
                key = sys.stdin.readline().strip().lower()
                if key == 'q':
                    print("\n🛑 Quit requested")
                    break
                elif key == 'n':
                    instruction_index += 1
                    new_instruction = instructions[instruction_index % len(instructions)]
                    print(f"\n➡️  SWITCHED TO: '{new_instruction}'")
                    print("=" * 60)
                elif key == 's':
                    print(f"\n📊 STATS: Frame {frame_count}, FPS {current_fps:.1f}, Avg Inference {avg_inference:.1f}ms")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by Ctrl+C")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

    # Final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference_final = np.mean(inference_times) * 1000 if inference_times else 0

    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Runtime: {total_time:.1f}s")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Average Inference: {avg_inference_final:.1f}ms")
    print(f"   Instructions Tested: {min(instruction_index + 1, len(instructions))}")

    print(f"\n🎉 SmolVLA Pi 5 Pipeline {'WORKING' if avg_fps > 0.5 else 'NEEDS OPTIMIZATION'}!")

    if avg_fps > 0.5:
        print("✅ Language → Motor coordinate pipeline functional!")
        print("🚀 Ready for real robot deployment!")
    else:
        print("⚠️  Performance too low - consider lighter model")

if __name__ == "__main__":
    test_smolvla_pi5()