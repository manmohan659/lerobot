import argparse
import random
import time

import cv2
import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait


def detect_close_legs(frame_bgr: np.ndarray, min_height_ratio: float = 0.45, bottom_band: float = 0.75) -> bool:
    """Return True if a person bbox indicates legs are close (tall and near bottom).

    min_height_ratio: fraction of image height the bbox must span to be considered close
    bottom_band: normalized y position (0..1) where bbox bottom must be below
    """
    H, W = frame_bgr.shape[:2]

    # Downscale for faster HOG detection
    target_w = 320
    scale = 1.0
    frame_small = frame_bgr
    if W > target_w:
        scale = target_w / float(W)
        frame_small = cv2.resize(frame_bgr, (int(W * scale), int(H * scale)))

    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:
        return False

    rects, _ = hog.detectMultiScale(frame_small, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(rects) == 0:
        return False

    for (x, y, w, h) in rects:
        # Rescale to original coordinates
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        height_ratio = h / float(H)
        bottom_y = (y + h) / float(H)
        if height_ratio >= min_height_ratio and bottom_y >= bottom_band:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="LeKiwi guard mode (no leader arm): reactive base + gripper (legs)")
    parser.add_argument("--ip", default="192.168.1.66", help="LeKiwi host IP")
    parser.add_argument("--id", default="my_awesome_kiwi", help="Robot id (client-side)")
    parser.add_argument("--fps", type=int, default=15, help="Control loop FPS")
    args = parser.parse_args()

    robot_cfg = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    robot = LeKiwiClient(robot_cfg)

    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to LeKiwi host.")

    try:
        print("Guard mode running. Ctrl+C to stop.")
        last_turn_change = time.perf_counter()
        current_turn = 0.0
        gripper_open = True

        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            front = obs.get("front", None)
            if isinstance(front, np.ndarray):
                frame = front
            else:
                frame = next((v for v in obs.values() if isinstance(v, np.ndarray)), None)

            # Decide action
            x_vel = 0.05
            y_vel = 0.0
            theta_vel = current_turn
            action = {}

            if frame is not None and detect_close_legs(frame):
                # Retreat and toggle gripper when a large face is near
                x_vel = -0.1
                theta_vel = 0.0
                action["arm_gripper.pos"] = 100.0 if gripper_open else 0.0
                gripper_open = not gripper_open

            # Randomly adjust heading every ~2 seconds during wander
            if time.perf_counter() - last_turn_change > 2.0 and x_vel > 0:
                current_turn = random.choice([0.0, 20.0, -20.0])
                theta_vel = current_turn
                last_turn_change = time.perf_counter()

            action.update({"x.vel": x_vel, "y.vel": y_vel, "theta.vel": theta_vel})
            robot.send_action(action)

            busy_wait(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("Stopping guard mode...")
    finally:
        try:
            # Stop base
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()


