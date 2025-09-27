import time
from typing import Dict

from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient


class Picker:
    def __init__(self, open_val: float = 20.0, close_val: float = 80.0, step_s: float = 0.3) -> None:
        self.open_val = open_val
        self.close_val = close_val
        self.step_s = step_s

    def _send(self, robot: LeKiwiClient, arm: Dict[str, float]) -> None:
        robot.send_action(arm)
        time.sleep(self.step_s)

    def run_pick(self, robot: LeKiwiClient) -> bool:
        # Open
        self._send(robot, {"arm_gripper.pos": self.open_val})
        # Descend (coarse, tune with your setup)
        self._send(
            robot,
            {
                "arm_shoulder_lift.pos": -5.0,
                "arm_elbow_flex.pos": 8.0,
            },
        )
        # Close
        self._send(robot, {"arm_gripper.pos": self.close_val})
        # Lift
        self._send(
            robot,
            {
                "arm_shoulder_lift.pos": 3.0,
                "arm_elbow_flex.pos": -6.0,
            },
        )
        return True


