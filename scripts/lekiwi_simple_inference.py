#!/usr/bin/env python3

"""
LeKiwi Simple Inference - Official LeRobot Way
Based on examples/lekiwi/evaluate.py but simplified for testing
"""

import time
import logging
import torch
import numpy as np
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.control_utils import predict_action, log_control_info, init_keyboard_listener
from lerobot.utils.utils import log_say

# Configuration
EPISODE_TIME_SEC = 60
FPS = 30
TASK_DESCRIPTION = "Pick up the red cup"

# Use your trained model (update this path)
YOUR_TRAINED_MODEL = "path/to/your/trained/model"  # Update this!

def main():
    """Main inference loop using official LeRobot approach"""

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Robot configuration - update IP for your LeKiwi
    robot_config = LeKiwiClientConfig(
        remote_ip="172.18.134.136",  # Update this IP!
        id="lekiwi"
    )
    robot = LeKiwiClient(robot_config)

    # Load your trained ACT policy
    try:
        # Try to load your trained model first
        policy = ACTPolicy.from_pretrained(YOUR_TRAINED_MODEL)
        log_say(f"✅ Loaded trained model: {YOUR_TRAINED_MODEL}")
    except Exception as e:
        # Fallback to a pretrained model for testing
        log_say(f"⚠️ Failed to load {YOUR_TRAINED_MODEL}: {e}")
        log_say("🔄 Loading pretrained model for testing...")
        policy = ACTPolicy.from_pretrained("lerobot/act_lekiwi_real")

    # Connect to robot
    log_say("🤖 Connecting to LeKiwi robot...")
    robot.connect()

    if not robot.is_connected:
        raise ValueError("❌ Robot is not connected!")

    log_say("✅ Robot connected successfully!")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_say(f"🖥️ Using device: {device}")

    policy = policy.to(device)
    policy.eval()

    # Initialize keyboard listener for control
    listener, events = init_keyboard_listener()

    log_say("🚀 Starting inference loop...")
    log_say("Controls: Right arrow = exit, Left arrow = restart, Esc = stop")
    log_say("=" * 60)

    # Main control loop
    dt_s = 1 / FPS
    episode_index = 0
    frame_index = 0

    start_time = time.perf_counter()

    try:
        while not events["stop_recording"]:
            start_loop = time.perf_counter()

            # Read current robot observation
            observation = robot.capture_observation()

            # Use official predict_action function
            action = predict_action(
                observation=observation,
                policy=policy,
                device=device,
                use_amp=False,  # No mixed precision for simplicity
                task=TASK_DESCRIPTION,
                robot_type=robot.robot_type
            )

            # Send action to robot
            robot.send_action(action)

            # Logging every 10 frames
            if frame_index % 10 == 0:
                elapsed_time = time.perf_counter() - start_time
                log_control_info(
                    robot=robot,
                    dt_s=time.perf_counter() - start_loop,
                    episode_index=episode_index,
                    frame_index=frame_index,
                    fps=FPS
                )

                # Print action values for debugging
                action_values = action.cpu().numpy()
                log_say(f"🎯 Action: [{', '.join(f'{x:6.3f}' for x in action_values[:6])}]")
                log_say(f"⏱️ Runtime: {elapsed_time:.1f}s, Frame: {frame_index}")

            frame_index += 1

            # Exit conditions
            if events["exit_early"]:
                log_say("⏹️ Early exit requested")
                break

            if frame_index * dt_s > EPISODE_TIME_SEC:
                log_say(f"✅ Episode completed ({EPISODE_TIME_SEC}s)")
                break

            # Sleep to maintain FPS
            sleep_time = dt_s - (time.perf_counter() - start_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log_say("🛑 Interrupted by user")

    except Exception as e:
        log_say(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        log_say("🧹 Cleaning up...")
        robot.disconnect()
        if listener:
            listener.stop()

        total_time = time.perf_counter() - start_time
        avg_fps = frame_index / total_time if total_time > 0 else 0

        log_say("📊 Final Stats:")
        log_say(f"   Total frames: {frame_index}")
        log_say(f"   Total time: {total_time:.1f}s")
        log_say(f"   Average FPS: {avg_fps:.1f}")
        log_say("🎉 Inference completed!")

if __name__ == "__main__":
    main()