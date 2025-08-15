# Model/annotations.py
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]

@dataclass(frozen=True)
class PanelAnnotations:
    seg: List[Point]
    points: List[Point]

@dataclass(frozen=True)
class AnnotationsBundle:
    highres: PanelAnnotations
    sd: PanelAnnotations
    micro: PanelAnnotations  # seg bleibt i. d. R. leer