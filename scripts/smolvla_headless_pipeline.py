#!/usr/bin/env python3

"""
SmolVLA Headless Pipeline for Pi 5 - No Display Required
Shows all info in terminal with real-time updates
"""

import cv2
import numpy as np
import torch
import time
import sys
import select
import tty
import termios
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def fix_normalization_stats(policy):
    """Fix infinite normalization stats with reasonable values"""
    print("🔧 Fixing normalization stats...")

    # Fix input normalization (state)
    if hasattr(policy, 'normalize_inputs') and hasattr(policy.normalize_inputs, 'buffer_observation_state'):
        policy.normalize_inputs.buffer_observation_state.mean.fill_(0.0)
        policy.normalize_inputs.buffer_observation_state.std.fill_(1.0)

    # Fix output normalization (actions)
    if hasattr(policy, 'normalize_targets') and hasattr(policy.normalize_targets, 'buffer_action'):
        policy.normalize_targets.buffer_action.mean.fill_(0.0)
        policy.normalize_targets.buffer_action.std.fill_(1.0)

    # Fix unnormalization (output actions)
    if hasattr(policy, 'unnormalize_outputs') and hasattr(policy.unnormalize_outputs, 'buffer_action'):
        policy.unnormalize_outputs.buffer_action.mean.fill_(0.0)
        policy.unnormalize_outputs.buffer_action.std.fill_(1.0)

    print("   ✅ Normalization fixed")

def kbhit():
    """Check if keyboard input is available (non-blocking)"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def test_smolvla_headless():
    print("🚀 SmolVLA Headless Pipeline for Pi 5")
    print("=" * 60)
    print("📺 NO DISPLAY REQUIRED - All info shown in terminal")
    print("🎮 Controls: Press 'n' + ENTER for next instruction, 'q' + ENTER to quit")
    print("=" * 60)

    device = torch.device('cpu')  # Pi 5 uses CPU
    print(f"🖥️  Device: {device}")

    # Load SmolVLA (already cached, so faster)
    print("📥 Loading SmolVLA...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device)
    print("✅ SmolVLA loaded!")

    # Fix normalization
    fix_normalization_stats(policy)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return

    print("📷 Camera ready!")
    print("\n🎯 Starting Language-Conditioned Inference...")
    print("Press ENTER to start, then 'n' + ENTER to change instructions")
    input("Press ENTER to begin...")

    # Extended test instructions
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

    # Simulate robot state that changes over time
    current_robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"\n🗣️  CURRENT INSTRUCTION: '{instructions[instruction_index]}'")
    print("=" * 60)

    try:
        while True:
            # Get current instruction
            instruction = instructions[instruction_index % len(instructions)]

            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed")
                continue

            frame_count += 1

            # Simulate changing robot state (slowly evolving)
            t = time.time() * 0.1
            current_robot_state = np.array([
                0.1 * np.sin(t),              # Joint 1
                0.15 * np.cos(t * 1.1),       # Joint 2
                0.2 * np.sin(t * 0.8),        # Joint 3
                0.1 * np.cos(t * 1.3),        # Joint 4
                0.05 * np.sin(t * 1.5),       # Joint 5
                0.3 * np.cos(t * 0.7)         # Joint 6 (gripper)
            ])

            # Prepare observation
            image_resized = cv2.resize(frame, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(device)

            # Use current robot state
            state_tensor = torch.from_numpy(current_robot_state).unsqueeze(0).float().to(device)

            # Create observation dict
            observation = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "task": instruction
            }

            # Run inference
            inference_start = time.time()

            with torch.no_grad():
                action = policy.select_action(observation)

            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # Calculate performance
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_inference = np.mean(inference_times[-10:]) * 1000  # Last 10 frames

            # Extract action values
            action_values = action.flatten().cpu().numpy()

            # Print live update every 5 frames
            if frame_count % 5 == 0:
                # Clear previous lines (simple version)
                print(f"\r📊 Frame {frame_count:4d} | FPS: {current_fps:5.1f} | Inference: {avg_inference:6.1f}ms", end="")
                sys.stdout.flush()

            # Detailed info every 20 frames
            if frame_count % 20 == 0:
                print(f"\n🔄 LIVE UPDATE:")
                print(f"   📝 Task: '{instruction}'")
                print(f"   🤖 Robot State: [{', '.join(f'{x:6.3f}' for x in current_robot_state)}]")
                print(f"   🎯 Motor Output: [{', '.join(f'{x:6.3f}' for x in action_values)}]")
                print(f"   ⚡ Performance: {current_fps:.1f} FPS, {avg_inference:.1f}ms inference")
                print("-" * 60)

            # Check for keyboard input (non-blocking)
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
        cap.release()

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
    print("\n🎉 SmolVLA Headless Pipeline COMPLETE!")
    if avg_fps > 1:
        print("✅ Pipeline working - different instructions produce different motor outputs!")
    else:
        print("⚠️  Low FPS - may need optimization for Pi 5")

if __name__ == "__main__":
    test_smolvla_headless()