#!/usr/bin/env python

import base64
import json
import io
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

try:
    import torch  # noqa: F401
    from PIL import Image
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "Detector server missing dependencies. Install with: pip install ultralytics fastapi uvicorn[standard] pillow"
    ) from e


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    label: str
    score: float
    bbox: BBox


class DetectRequest(BaseModel):
    image_jpeg_base64: str
    labels: Optional[List[str]] = None


class DetectResponse(BaseModel):
    detections: List[Detection]
    latency_ms: float


def _load_model() -> YOLO:
    model_name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    model = YOLO(model_name)
    return model


MODEL: YOLO = _load_model()
APP = FastAPI()


def _log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        print(base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8"))
    except Exception:
        print(str(payload))


@APP.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@APP.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest, request: Request) -> Any:
    t0 = time.perf_counter()
    try:
        img_bytes = base64.b64decode(req.image_jpeg_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run YOLO inference
    imgsz_env = os.environ.get("YOLO_IMGSZ", "640")
    try:
        if "," in imgsz_env or "x" in imgsz_env.lower():
            parts = imgsz_env.lower().replace("x", ",").split(",")
            imgsz = [int(parts[0]), int(parts[1])] if len(parts) >= 2 else int(parts[0])
        else:
            imgsz = int(imgsz_env)
    except Exception:
        imgsz = 640

    results = MODEL.predict(img, imgsz=imgsz, verbose=False)
    detections: List[Detection] = []

    # Collect detections from first result
    if len(results) > 0:
        r = results[0]
        names = r.names
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_idx = int(b.cls.item())
                label = names.get(cls_idx, str(cls_idx))
                score = float(b.conf.item())
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                detections.append(
                    Detection(label=label, score=score, bbox=BBox(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1)))
                )

    # If label filter provided, filter; else return top few
    if req.labels:
        allowed = {lbl.lower() for lbl in req.labels}
        detections = [d for d in detections if d.label.lower() in allowed]

    # Fallback: keep top-3 by score if many
    detections.sort(key=lambda d: d.score, reverse=True)
    if len(detections) > 5:
        detections = detections[:5]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    _log(
        "detect",
        remote=str(request.client.host if request.client else "-"),
        num_dets=len(detections),
        latency_ms=latency_ms,
    )
    return DetectResponse(detections=detections, latency_ms=latency_ms)


app = APP  # uvicorn entrypoint


