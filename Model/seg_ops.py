# Model/seg_ops.py
from __future__ import annotations
import math
from math import cos, pi
from typing import Sequence, Tuple, List

Point = Tuple[float, float]

def deform_points_gaussian(
    pts: List[Point],
    center: Point,
    delta: Point,            # (dx, dy)
    radius: float,           # in Bild-Pixeln
    strength: float = 1.0,
    sigma: float | None = None
) -> None:
    """
    Verschiebt Punkte innerhalb 'radius' um center mit gaußgewichtetem Anteil.
    Modifiziert pts IN PLACE.
    """
    if not pts or radius <= 0:
        return
    if sigma is None:
        sigma = max(1.0, 0.5 * radius)
    R2 = radius * radius
    two_sigma2 = 2.0 * sigma * sigma
    cx, cy = center
    dx, dy = delta

    for i, (px, py) in enumerate(pts):
        d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy)
        if d2 > R2:
            continue
        w = math.exp(-d2 / two_sigma2) * strength
        pts[i] = (px + w * dx, py + w * dy)

def laplacian_smooth(pts: List[Point], iters: int = 1, lam: float = 0.2) -> None:
    """Einfache Laplace-Glättung, IN PLACE."""
    n = len(pts)
    if n < 3 or iters <= 0 or lam <= 0:
        return
    for _ in range(iters):
        new = pts[:]
        for i in range(1, n - 1):
            ax, ay = pts[i - 1]
            bx, by = pts[i + 1]
            cx, cy = pts[i]
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            new[i] = (cx + lam * (mx - cx), cy + lam * (my - cy))
        pts[:] = new

def _euclid(a: Point, b: Point) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return (dx*dx + dy*dy) ** 0.5

def nearest_seg_point_index(pts: Sequence[Point], x: float, y: float) -> tuple[int|None, float]:
    best_i, best_d2 = None, float("inf")
    for i, (px, py) in enumerate(pts):
        d2 = (px - x)*(px - x) + (py - y)*(py - y)
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i, best_d2 ** 0.5

def build_edit_window(pts: Sequence[Point], idx_center: int, radius_px: float) -> tuple[list[int], list[float]]:
    n = len(pts)
    indices = {idx_center}
    dist_from_center = {idx_center: 0.0}

    def d(i, j): return _euclid(pts[i], pts[j])

    # links
    dist, j = 0.0, idx_center
    while j > 0:
        dist += d(j, j-1)
        if dist > radius_px: break
        j -= 1
        indices.add(j); dist_from_center[j] = dist

    # rechts
    dist, j = 0.0, idx_center
    while j < n - 1:
        dist += d(j, j+1)
        if dist > radius_px: break
        j += 1
        indices.add(j); dist_from_center[j] = dist

    idx_list = sorted(indices)
    w_list = []
    for j in idx_list:
        s = min(dist_from_center.get(j, 0.0), radius_px)
        w = 0.5 * (1.0 + cos(pi * (s / radius_px)))  # 1 → 0
        w_list.append(w)
    return idx_list, w_list