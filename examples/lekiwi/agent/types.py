from dataclasses import dataclass
from typing import Optional


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Detection:
    label: str
    score: float
    bbox: BBox


@dataclass
class Intent:
    task: str  # e.g., "fetch"
    object: str  # e.g., "tissue"
    constraints: tuple[str, ...] = ()
    confirm: bool = True


ActionDict = dict[str, float]


