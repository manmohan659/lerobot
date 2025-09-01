import argparse
import time
from typing import Any

import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def try_imports() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import cv2  # type: ignore

        versions["cv2"] = getattr(cv2, "__version__", "unknown")
    except Exception as e:  # pragma: no cover
        versions["cv2"] = f"MISSING ({e})"
    try:
        import numpy  # type: ignore

        versions["numpy"] = getattr(numpy, "__version__", "unknown")
    except Exception as e:  # pragma: no cover
        versions["numpy"] = f"MISSING ({e})"
    return versions


def wait_observation(robot: LeKiwiClient, timeout_s: float = 5.0) -> dict[str, Any]:
    start = time.perf_counter()
    obs: dict[str, Any] | None = None
    while time.perf_counter() - start < timeout_s:
        obs = robot.get_observation()
        if obs:
            return obs
        time.sleep(0.05)
    raise TimeoutError("No observation received from host within timeout")


def check_cameras(obs: dict[str, Any]) -> list[str]:
    found = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            h, w, c = v.shape
            if c == 3 and h > 0 and w > 0:
                found.append(f"{k} {h}x{w}")
    return found


def base_motion_test(robot: LeKiwiClient, dwell_s: float = 0.6) -> list[dict[str, float]]:
    seq = [
        {"x.vel": 0.10, "y.vel": 0.00, "theta.vel": 0.0},
        {"x.vel": -0.10, "y.vel": 0.00, "theta.vel": 0.0},
        {"x.vel": 0.00, "y.vel": 0.10, "theta.vel": 0.0},
        {"x.vel": 0.00, "y.vel": -0.10, "theta.vel": 0.0},
        {"x.vel": 0.00, "y.vel": 0.00, "theta.vel": 30.0},
        {"x.vel": 0.00, "y.vel": 0.00, "theta.vel": -30.0},
    ]
    samples: list[dict[str, float]] = []
    for cmd in seq:
        robot.send_action(cmd)
        time.sleep(dwell_s)
        obs = robot.get_observation()
        samples.append({
            "x.vel": float(obs.get("x.vel", 0.0)),
            "y.vel": float(obs.get("y.vel", 0.0)),
            "theta.vel": float(obs.get("theta.vel", 0.0)),
        })
    # stop
    robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    return samples


def arm_motion_test(robot: LeKiwiClient, dwell_s: float = 0.7) -> dict[str, tuple[float, float]]:
    joints = [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
    ]
    obs0 = robot.get_observation()
    deltas: dict[str, float] = {}
    for j in joints:
        cur = float(obs0.get(j, 0.0))
        if j.endswith("gripper.pos"):
            tgt = clamp(cur + 10.0, 0.0, 100.0)
        else:
            tgt = clamp(cur + 5.0, -100.0, 100.0)
        robot.send_action({j: tgt, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        time.sleep(dwell_s)
        obs1 = robot.get_observation()
        new = float(obs1.get(j, cur))
        deltas[j] = new - cur
        # return to original
        robot.send_action({j: cur})
        time.sleep(0.4)
    return {j: (float(obs0.get(j, 0.0)), float(obs0.get(j, 0.0)) + deltas[j]) for j in joints}


def main():
    p = argparse.ArgumentParser("LeKiwi guard-mode preflight test (no leader arm)")
    p.add_argument("--ip", default="192.168.1.66")
    p.add_argument("--id", default="my_awesome_kiwi")
    p.add_argument("--skip-cameras", action="store_true")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-arm", action="store_true")
    args = p.parse_args()

    versions = try_imports()
    print("Dependencies (laptop):", versions)

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.ip, id=args.id))
    robot.connect()

    try:
        obs = wait_observation(robot)
        print("Connected: received observation with keys:", list(obs.keys())[:8], "...")

        if not args.skip_cameras:
            cams = check_cameras(obs)
            print("Cameras:", cams if cams else "none found in observation")

        if not args.skip_base:
            samples = base_motion_test(robot)
            print("Base velocity samples:", samples)

        if not args.skip_arm:
            moved = arm_motion_test(robot)
            print("Arm test deltas (approx):", moved)

        print("Preflight OK. You can run guard_mode now.")
    finally:
        try:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()


