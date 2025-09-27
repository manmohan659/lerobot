#!/usr/bin/env python3

"""
LeKiwi Real Control - Vision to Action with Real Robot
Combines proven vision pipeline with real LeKiwi robot control
"""

import torch
import numpy as np
import time
import sys
import select
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.control_utils import predict_action, log_control_info

def can_display():
    """Check if we can show windows"""
    import os
    return 'DISPLAY' in os.environ

def kbhit():
    """Non-blocking keyboard input check"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def convert_aloha_to_lekiwi_action(aloha_action: np.ndarray) -> dict[str, float]:
    """
    Convert ALOHA 14D action to LeKiwi action format

    ALOHA: 14D dual arm [arm1_joints(7) + arm2_joints(7)]
    LeKiwi: arm(6) + base_motion(3) = 9D total

    LeKiwi state order (from config):
    - arm_shoulder_pan.pos
    - arm_shoulder_lift.pos
    - arm_elbow_flex.pos
    - arm_wrist_flex.pos
    - arm_wrist_roll.pos
    - arm_gripper.pos
    - x.vel
    - y.vel
    - theta.vel
    """

    # Take first 6 joints from ALOHA (ignore second arm)
    arm_positions = aloha_action[:6]

    # Simple base motion - derive from arm motion patterns
    # You can tune these mappings based on your task
    base_x = np.clip(aloha_action[1] * 0.1, -0.2, 0.2)  # Forward/back based on shoulder lift
    base_y = np.clip(aloha_action[0] * 0.1, -0.2, 0.2)  # Left/right based on shoulder pan
    base_theta = np.clip(aloha_action[4] * 20, -45, 45)  # Rotation based on wrist roll

    # Create LeKiwi action dict
    lekiwi_action = {
        "arm_shoulder_pan.pos": float(arm_positions[0]),
        "arm_shoulder_lift.pos": float(arm_positions[1]),
        "arm_elbow_flex.pos": float(arm_positions[2]),
        "arm_wrist_flex.pos": float(arm_positions[3]),
        "arm_wrist_roll.pos": float(arm_positions[4]),
        "arm_gripper.pos": float(arm_positions[5]),
        "x.vel": float(base_x),
        "y.vel": float(base_y),
        "theta.vel": float(base_theta),
    }

    return lekiwi_action

def create_aloha_observation(robot_obs: dict) -> dict:
    """
    Convert LeKiwi observation to ALOHA format for the ACT model
    """

    # Get robot camera (use 'front' camera from LeKiwi)
    if "front" in robot_obs:
        camera_image = robot_obs["front"]

        # Resize to ALOHA expected size (480, 640) if needed
        if camera_image.shape[:2] != (480, 640):
            import cv2
            camera_image = cv2.resize(camera_image, (640, 480))

        # Convert BGR to RGB (OpenCV uses BGR)
        import cv2
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        # Convert to tensor: HWC -> CHW, uint8 -> float32 in [0,1]
        image_tensor = torch.from_numpy(camera_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    else:
        # Fallback: create dummy image
        print("⚠️ No 'front' camera found, using dummy image")
        image_tensor = torch.zeros(1, 3, 480, 640)

    # Get robot state and convert to ALOHA format (14D)
    if "observation.state" in robot_obs:
        lekiwi_state = robot_obs["observation.state"]  # 9D LeKiwi state

        # Pad to 14D for ALOHA (duplicate arm for dual-arm compatibility)
        if len(lekiwi_state) == 9:
            # Take arm positions (first 6) and duplicate for second arm
            arm_state = lekiwi_state[:6]
            aloha_state = np.concatenate([arm_state, arm_state, lekiwi_state[6:9][:2]])  # 6+6+2=14
        else:
            # Fallback: pad or truncate to 14
            aloha_state = np.zeros(14, dtype=np.float32)
            aloha_state[:min(len(lekiwi_state), 14)] = lekiwi_state[:14]
    else:
        # Fallback: dummy state
        print("⚠️ No robot state found, using dummy state")
        aloha_state = np.zeros(14, dtype=np.float32)

    state_tensor = torch.from_numpy(aloha_state).unsqueeze(0)  # Add batch dimension

    # Create ALOHA observation
    aloha_observation = {
        "observation.images.top": image_tensor,
        "observation.state": state_tensor,
    }

    return aloha_observation

def main():
    print("🚀 LeKiwi Real Control - Vision to Action")
    print("=" * 60)

    # Configuration - UPDATE THIS IP!
    ROBOT_IP = "192.168.1.100"  # ⚠️ UPDATE THIS TO YOUR LEKIWI IP!
    EPISODE_TIME_SEC = 60
    FPS = 10  # Lower FPS for real robot safety

    print(f"🤖 Connecting to LeKiwi at {ROBOT_IP}")
    print("⚠️  Make sure LeKiwi host is running on the robot!")

    # Check display
    has_display = can_display()
    print(f"🖥️ Display: {'Available' if has_display else 'Headless'}")

    # Configure robot
    robot_config = LeKiwiClientConfig(
        remote_ip=ROBOT_IP,
        id="lekiwi_vision_control"
    )
    robot = LeKiwiClient(robot_config)

    # Load ACT model
    print("📥 Loading ACT model...")
    policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

    device = torch.device("cpu")
    policy = policy.to(device)
    policy.eval()
    policy.reset()  # Important!

    print("✅ ACT model loaded and ready")

    # Connect to robot
    try:
        print("🔌 Connecting to LeKiwi robot...")
        robot.connect()
        print("✅ Robot connected successfully!")
    except Exception as e:
        print(f"❌ Robot connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Is LeKiwi host running? Run on robot: python -m lerobot.robots.lekiwi.lekiwi_host")
        print(f"   2. Is IP correct? Currently: {ROBOT_IP}")
        print("   3. Is network connection working? Try: ping {ROBOT_IP}")
        return

    print("\n🎯 Starting vision-action control...")
    print(f"Controls: 'q' + ENTER = quit, 's' + ENTER = stop robot")
    print(f"Episode duration: {EPISODE_TIME_SEC}s")
    print("=" * 60)

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    inference_times = []
    robot_stopped = False

    dt_s = 1 / FPS

    try:
        while True:
            loop_start = time.time()

            # Get robot observation
            try:
                robot_obs = robot.get_observation()
            except Exception as e:
                print(f"❌ Failed to get robot observation: {e}")
                continue

            # Convert to ALOHA format for model
            aloha_obs = create_aloha_observation(robot_obs)

            # Model inference
            inference_start = time.time()
            with torch.inference_mode():
                aloha_action = policy.select_action(aloha_obs)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # Convert ALOHA action to LeKiwi action
            aloha_action_np = aloha_action.squeeze(0).cpu().numpy()
            lekiwi_action = convert_aloha_to_lekiwi_action(aloha_action_np)

            # Send action to robot (unless stopped)
            if not robot_stopped:
                try:
                    robot.send_action(lekiwi_action)
                except Exception as e:
                    print(f"❌ Failed to send action: {e}")

            frame_count += 1

            # Performance stats
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_inference = np.mean(inference_times[-10:]) * 1000

            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:4d} | FPS: {current_fps:4.1f} | Inference: {avg_inference:5.1f}ms | {'STOPPED' if robot_stopped else 'ACTIVE'}")
                print(f"   Robot State: {robot_obs.get('observation.state', 'N/A')}")
                print(f"   Action: arm=[{', '.join(f'{lekiwi_action[k]:.2f}' for k in ['arm_shoulder_pan.pos', 'arm_shoulder_lift.pos', 'arm_elbow_flex.pos'])}] base=[{lekiwi_action['x.vel']:.2f}, {lekiwi_action['y.vel']:.2f}, {lekiwi_action['theta.vel']:.1f}]")
                print("-" * 60)

            # Control flow via logging
            log_control_info(
                robot=robot,
                dt_s=time.time() - loop_start,
                episode_index=0,
                frame_index=frame_count,
                fps=FPS
            )

            # Check for quit/stop commands
            if kbhit():
                key = sys.stdin.readline().strip().lower()
                if key == 'q':
                    print("\n🛑 Quit requested")
                    break
                elif key == 's':
                    robot_stopped = not robot_stopped
                    status = "STOPPED" if robot_stopped else "RESUMED"
                    print(f"\n⏸️  Robot {status}")

            # Episode time limit
            if elapsed > EPISODE_TIME_SEC:
                print(f"\n⏰ Episode completed ({EPISODE_TIME_SEC}s)")
                break

            # Sleep to maintain FPS
            sleep_time = dt_s - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Keep lists manageable
            if len(inference_times) > 100:
                inference_times = inference_times[-50:]

    except KeyboardInterrupt:
        print("\n🛑 Stopped by Ctrl+C")

    finally:
        # Stop robot safely
        print("🛑 Stopping robot...")
        try:
            # Send zero action to stop
            zero_action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": 0.0,
                "arm_elbow_flex.pos": 0.0,
                "arm_wrist_flex.pos": 0.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 0.0,
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
            robot.send_action(zero_action)
            time.sleep(0.1)  # Give time for command to reach robot
        except Exception as e:
            print(f"⚠️ Error stopping robot: {e}")

        # Disconnect
        try:
            robot.disconnect()
            print("✅ Robot disconnected safely")
        except Exception as e:
            print(f"⚠️ Error disconnecting: {e}")

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
            print("🎉 EXCELLENT - Vision-action pipeline working!")
        else:
            print("⚠️ Performance could be improved")

        print("\n✅ Session completed safely!")

if __name__ == "__main__":
    main()