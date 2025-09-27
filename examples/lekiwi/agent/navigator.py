from typing import List

import numpy as np

from .types import ActionDict, Detection


class Navigator:
    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        k_lat: float = 0.5,
        k_yaw: float = 60.0,
        k_fwd: float = 1.0,
        v_lat: float = 0.2,
        v_yaw: float = 45.0,
        v_max_far: float = 0.25,
        v_max_near: float = 0.08,
        target_area_ratio: float = 0.06,
        near_thresh: float = 0.04,
        scan_rate_deg: float = 20.0,
    ) -> None:
        self.W = img_width
        self.H = img_height
        self.k_lat = k_lat
        self.k_yaw = k_yaw
        self.k_fwd = k_fwd
        self.v_lat = v_lat
        self.v_yaw = v_yaw
        self.v_max_far = v_max_far
        self.v_max_near = v_max_near
        self.target_area_ratio = target_area_ratio
        self.near_thresh = near_thresh
        self.scan_rate_deg = scan_rate_deg
        self._scan_dir = 1.0

    def compute_action(self, detections: List[Detection]) -> ActionDict:
        if len(detections) == 0:
            # Slow scan in place
            theta = self.scan_rate_deg * self._scan_dir
            self._scan_dir *= -1.0
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": theta}

        # Use top detection
        det = max(detections, key=lambda d: d.score)
        cx = det.bbox.x + det.bbox.w / 2.0
        area_ratio = (det.bbox.w * det.bbox.h) / float(self.W * self.H)
        ex = (cx - self.W / 2.0) / float(self.W)

        y = np.clip(-self.k_lat * ex, -self.v_lat, self.v_lat)
        theta = np.clip(-self.k_yaw * ex, -self.v_yaw, self.v_yaw)
        fwd_cmd = self.k_fwd * (self.target_area_ratio - area_ratio)
        fwd_cmd = np.clip(fwd_cmd, 0.0, self.v_max_far)
        if area_ratio > self.near_thresh:
            fwd_cmd = min(fwd_cmd, self.v_max_near)

        return {"x.vel": float(fwd_cmd), "y.vel": float(y), "theta.vel": float(theta)}


