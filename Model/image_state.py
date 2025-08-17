from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import numpy as np
    NdArray = np.ndarray
except Exception:
    NdArray = object

Point = tuple[float, float]
PointList = list[Point]

@dataclass
class ImageState:
    path: Optional[str] = None
    original: Optional["QImage"] = None      # <- wird auf das CROPPED-Bild gesetzt
    contrast: int = 150
    brightness: int = 0
    version: int = 0
    seg_points: list[tuple[float, float]] = field(default_factory=list)
    pts_points: list[tuple[float, float]] = field(default_factory=list)

    # NEU:
    crop_rect: Optional[tuple[int, int, int, int]] = None  # (x_start, y_start, x_end, y_end)
    rgb_np: Optional[NdArray] = None                       # optional: RGB-Array des Crops
