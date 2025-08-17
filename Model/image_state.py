# Model/image_state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtGui import QImage

Point = tuple[float, float]
PointList = list[Point]

@dataclass
class ImageState:
    path: Optional[str] = None
    original: Optional[QImage] = None
    contrast: int = 150
    brightness: int = 0
    version: int = 0
    seg_points: PointList = field(default_factory=list)
    pts_points: PointList = field(default_factory=list)
