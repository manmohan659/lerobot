import base64
import json
from typing import List

import numpy as np
import requests

from .types import BBox, Detection


class DetectorClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _encode_jpeg(self, frame_bgr: np.ndarray, quality: int = 70) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf).decode("utf-8")

    def detect(self, frame_bgr: np.ndarray, labels: List[str]) -> List[Detection]:
        # Downscale for bandwidth
        import cv2

        h, w = frame_bgr.shape[:2]
        scale = 320.0 / max(h, w)
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

        payload = {
            "image_jpeg_base64": self._encode_jpeg(frame_bgr, 70),
            "labels": labels,
        }
        resp = requests.post(f"{self.base_url}/detect", data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        dets: List[Detection] = []
        for d in data.get("detections", []):
            bb = d["bbox"]
            dets.append(Detection(label=d["label"], score=float(d["score"]), bbox=BBox(x=int(bb["x"]), y=int(bb["y"]), w=int(bb["w"]), h=int(bb["h"]))))
        return dets


