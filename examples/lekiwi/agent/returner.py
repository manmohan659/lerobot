import time
from typing import Optional

from .types import ActionDict


class Returner:
    def __init__(self) -> None:
        self._home_heading_deg: Optional[float] = None

    def set_home_heading(self, heading_deg: float) -> None:
        self._home_heading_deg = heading_deg

    def step(self, current_heading_deg: float) -> ActionDict:
        if self._home_heading_deg is None:
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        heading_error = self._home_heading_deg - current_heading_deg
        heading_error = (heading_error + 180.0) % 360.0 - 180.0
        theta_cmd = max(min(heading_error * 1.0, 30.0), -30.0)
        x_cmd = 0.15
        return {"x.vel": x_cmd, "y.vel": 0.0, "theta.vel": theta_cmd}


