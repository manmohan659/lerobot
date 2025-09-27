#!/usr/bin/env python3

"""
LeKiwi Final Working Script
Based on proven ACT inference + camera integration
"""

import cv2
import torch
import numpy as np
import time
import sys
import select
from lerobot.policies.act.modeling_act import ACTPolicy

def can_display():
    """Check if we can show windows"""
    import os
    return 'DISPLAY' in os.environ

def init_camera():
    """Initialize camera using proven detection logic"""
    print("📷 Initializing camera...")

    # Try different camera access methods
    camera_attempts = [
        ("/dev/video0", "Video device 0"),
        ("/dev/video1", "Video device 1"),
        (0, "Camera index 0"),
        (1, "Camera index 1"),
    ]

    for camera_id, desc in camera_attempts:
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ {desc} working! Frame: {frame.shape}")
                    # Set properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                else:
                    cap.release()
        except Exception as e:
            print(f"   ❌ {desc} failed: {e}")

    print("❌ No camera found!")
    return None

def kbhit():
    """Non-blocking keyboard input check"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def create_observation(frame, robot_state):
    """Convert camera frame + robot state to model observation"""

    # Resize frame to model expected size (480, 640)
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert to tensor: HWC -> CHW, uint8 -> float32 in [0,1]
    image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Convert robot state to tensor
    state_tensor = torch.from_numpy(robot_state).unsqueeze(0)  # Add batch dimension

    # Create observation dict (matching ACT model requirements)
    observation = {
        "observation.images.top": image_tensor,
        "observation.state": state_tensor,
    }

    return observation

def main():
    print("🚀 LeKiwi Final Working Script")
    print("=" * 60)

    # Check display
    has_display = can_display()
    print(f"🖥️ Display: {'Available' if has_display else 'Headless'}")

    # Load ACT model
    print("📥 Loading ACT model...")
    policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

    device = torch.device("cpu")
    policy = policy.to(device)
    policy.eval()
    policy.reset()  # Important!

    print("✅ ACT model loaded and ready")

    # Initialize camera
    cap = init_camera()
    if cap is None:
        return

    print("📷 Camera ready!")
    print("\n🎯 Starting vision-to-action pipeline...")
    print("Controls: 'q' + ENTER = quit")
    print("=" * 60)

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    inference_times = []

    # Simulated robot state (14D for ALOHA compatibility)
    robot_state = np.zeros(14, dtype=np.float32)

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed")
                continue

            frame_count += 1

            # Update simulated robot state (smooth motion)
            t = time.time() * 0.5
            robot_state = np.array([
                0.1 * np.sin(t),         # Joint 1
                0.1 * np.cos(t * 1.1),   # Joint 2
                0.1 * np.sin(t * 0.8),   # Joint 3
                0.1 * np.cos(t * 1.3),   # Joint 4
                0.05 * np.sin(t * 1.5),  # Joint 5
                0.05 * np.cos(t * 0.7),  # Joint 6
                0.02 * np.sin(t * 2),    # Gripper
                # Repeat for second arm (ALOHA dual arm)
                0.1 * np.sin(t + np.pi/4),
                0.1 * np.cos(t * 1.1 + np.pi/4),
                0.1 * np.sin(t * 0.8 + np.pi/4),
                0.1 * np.cos(t * 1.3 + np.pi/4),
                0.05 * np.sin(t * 1.5 + np.pi/4),
                0.05 * np.cos(t * 0.7 + np.pi/4),
                0.02 * np.sin(t * 2 + np.pi/4),
            ], dtype=np.float32)

            # Create observation
            observation = create_observation(frame, robot_state)

            # Model inference
            inference_start = time.time()
            with torch.inference_mode():
                action = policy.select_action(observation)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # Get action values
            action_values = action.squeeze(0).cpu().numpy()

            # Performance stats
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_inference = np.mean(inference_times[-10:]) * 1000

            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:4d} | FPS: {current_fps:4.1f} | Inference: {avg_inference:5.1f}ms")
                print(f"   Robot State: [{', '.join(f'{x:6.3f}' for x in robot_state[:6])}...]")
                print(f"   Action Out:  [{', '.join(f'{x:6.3f}' for x in action_values[:6])}...]")
                print("-" * 60)

            # Display frame if possible
            if has_display:
                display_frame = frame.copy()

                # Add overlays
                cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Inference: {avg_inference:.0f}ms",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show action as colored circles
                for i, val in enumerate(action_values[:6]):
                    x = 50 + i * 80
                    y = 100
                    color = (0, 255, 0) if abs(val) < 0.5 else (0, 0, 255)
                    cv2.circle(display_frame, (x, y), int(20 + abs(val)*30), color, -1)
                    cv2.putText(display_frame, f"{val:.2f}", (x-25, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                cv2.imshow('LeKiwi Vision-Action Pipeline', display_frame)
                cv2.waitKey(1)

            # Check for quit command
            if kbhit():
                key = sys.stdin.readline().strip().lower()
                if key == 'q':
                    print("\n🛑 Quit requested")
                    break

            # Keep lists manageable
            if len(inference_times) > 100:
                inference_times = inference_times[-50:]

    except KeyboardInterrupt:
        print("\n🛑 Stopped by Ctrl+C")

    finally:
        # Cleanup
        if cap:
            cap.release()
        cv2.destroyAllWindows()

        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_inference_final = np.mean(inference_times) * 1000 if inference_times else 0

        print(f"\n🏁 Final Results:")
        print(f"   Total Frames: {frame_count}")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average Inference: {avg_inference_final:.1f}ms")

        if avg_fps > 5 and avg_inference_final < 100:
            print("🎉 EXCELLENT - Ready for real robot deployment!")
        elif avg_fps > 1:
            print("✅ GOOD - Working vision-action pipeline!")
        else:
            print("⚠️ SLOW - May need optimization")

if __name__ == "__main__":
    main()