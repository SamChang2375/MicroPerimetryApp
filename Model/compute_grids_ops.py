# Model/compute_grids.py
from __future__ import annotations
from PyQt6.QtGui import QImage
from typing import List, Tuple
import math
import itertools
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

# Alternative Lösung mit Konturerkennung
def extract_segmentation_line(
        image_rgb: np.ndarray,
        target_rgb: tuple[int, int, int] = (0, 255, 255),  # Cyan
        tol: int = 10,
        min_pixels: int = 20
) -> list[tuple[float, float]]:
    if image_rgb is None:
        return []

    # Farb-Schwelle
    lo = np.array([max(0, target_rgb[0] - tol),
                   max(0, target_rgb[1] - tol),
                   max(0, target_rgb[2] - tol)], dtype=np.uint8)
    hi = np.array([min(255, target_rgb[0] + tol),
                   min(255, target_rgb[1] + tol),
                   min(255, target_rgb[2] + tol)], dtype=np.uint8)

    mask = cv2.inRange(image_rgb, lo, hi)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Find longest contour
    longest_contour = max(contours, key=lambda x: cv2.arcLength(x, False))

    # Convert contour points to list of tuples
    pts = [(float(pt[0][0]), float(pt[0][1])) for pt in longest_contour]

    return pts

Pt = Tuple[float, float]


def correct_segmentation(
        pts,
        tol: float = 1.5,
        auto_close: bool = True,
        max_step: float = 1.0,
        min_index_gap: int = 10,
):
    """
    Verbesserte Logik:
    1. Wenn die Linie bereits geschlossen ist (Anfang ~ Ende), wird sie unverändert zurückgegeben
    2. Andernfalls wird versucht, die Linie zu schließen:
       - Finde den Punkt, der dem ersten Punkt am nächsten ist (mit min_index_gap)
       - Schneide überhängende Teile ab
       - Verbinde die Enden mit einer geraden Linie
    """
    import math

    if not pts or len(pts) < 2:
        return None if not auto_close else list(pts)

    def peq(a, b) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol

    def interpolate(p, q, step: float):
        dx, dy = q[0] - p[0], q[1] - p[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return [q]
        n = max(1, int(math.ceil(dist / step)))
        return [(p[0] + dx * (i / n), p[1] + dy * (i / n)) for i in range(1, n + 1)]

    # Prüfe ob die Linie bereits geschlossen ist
    if len(pts) >= 2 and peq(pts[0], pts[-1]):
        return list(pts)

    if not auto_close:
        return None

    # Finde den besten Endpunkt zum Schließen
    if len(pts) < min_index_gap + 1:
        # Linie zu kurz - einfach direkt verbinden
        closed = list(pts)
        closed.extend(interpolate(closed[-1], closed[0], max_step))
        if not peq(closed[-1], closed[0]):
            closed.append(closed[0])
        return closed

    # Suche den besten Punkt zum Schließen
    first_pt = pts[0]
    best_dist = float('inf')
    best_idx = -1

    for i in range(min_index_gap, len(pts)):
        dist = math.hypot(first_pt[0] - pts[i][0], first_pt[1] - pts[i][1])
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx == -1 or best_dist > tol * 3:  # Großzügigere Toleranz fürs Schließen
        # Kein guter Punkt gefunden - einfach direkt verbinden
        closed = list(pts)
        closed.extend(interpolate(closed[-1], closed[0], max_step))
        if not peq(closed[-1], closed[0]):
            closed.append(closed[0])
        return closed

    # Schneide überhängende Teile ab und behalte den Hauptteil
    core = list(pts[:best_idx + 1])

    # Optional: Glätten der Verbindung
    if len(core) >= 2 and not peq(core[0], core[-1]):
        core.extend(interpolate(core[-1], core[0], max_step))
        if not peq(core[-1], core[0]):
            core.append(core[0])

    return core

Point = Tuple[float, float]

# Punktezuordnung über euklidische Distanzen für HR zu SD
def convert_np_floats_to_tuples(points: List) -> List[Tuple[float, float]]:
    """Hilfsfunktion um np.float64 Werte in normale Python floats umzuwandeln"""
    return [(float(x), float(y)) for x, y in points]


def order_points_by_min_distance(
        pts1: List[Tuple[float, float]],
        pts2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Ordnet die zweite Punktliste basierend auf minimaler euklidischer Distanz zur ersten Liste.

    Args:
        pts1: Referenzpunktliste (bleibt unverändert)
        pts2: Zu ordnende Punktliste (kann np.float64 Werte enthalten)

    Returns:
        Geordnete Version von pts2 mit normalen Python floats
    """
    # Konvertiere np.float64 Werte zu normalen floats
    pts1 = convert_np_floats_to_tuples(pts1)
    pts2 = convert_np_floats_to_tuples(pts2)

    if len(pts1) != len(pts2):
        raise ValueError("Beide Punktlisten müssen gleich lang sein")

    pts1_arr = np.array(pts1)
    pts2_arr = np.array(pts2)
    ordered_pts2 = []
    used_indices = set()

    for point1 in pts1_arr:
        min_dist = float('inf')
        best_idx = -1
        best_point = None

        # Finde den nächstgelegenen noch nicht verwendeten Punkt
        for idx, point2 in enumerate(pts2_arr):
            if idx in used_indices:
                continue

            dist = np.linalg.norm(point1 - point2)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                best_point = point2

        if best_idx == -1:
            raise RuntimeError("Kein passender Punkt gefunden")

        used_indices.add(best_idx)
        ordered_pts2.append(tuple(map(float, best_point)))  # Sicherstellen dass es normale floats sind

    return ordered_pts2

Point = Tuple[float, float]
PointList = List[Point]
def sort_points_via_center_overlay(
    sd_img_cropped: np.ndarray,
    mp_img: np.ndarray,
    sd_pts_points: PointList,        # bereits geordnet
    mp_pts_points_unordered: PointList  # ungeordnet
) -> PointList:
    """
    Pipeline:
      1) SD-OCT-Bild auf fx=0.778, fy=0.801 skalieren -> resized_sd
         und SD-Punkteliste identisch skalieren.
      2) Beide Bilder werden über ihr Zentrum ausgerichtet (nur Translation).
         => MP-Punkte in das Koordinatensystem des 'resized_sd' transformieren.
         -> mp_to_sd_unordered
      3) Sortieren: mp_to_sd_ordered = order_points_by_min_distance(sd_pts_scaled, mp_to_sd_unordered)
      4) mp_to_sd_ordered zurück ins MP-Koordinatensystem transformieren.
         -> sd_to_mp_ordered
      5) Rückgabe: sd_to_mp_ordered
    """

    # --- 1) SD-Bild + SD-Punkte skalieren ---
    fx, fy = 0.778, 0.801
    # Bild skalieren
    resized_sd = cv2.resize(sd_img_cropped, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    # SD-Punkte skalieren (x*fx, y*fy)
    sd_pts_scaled: PointList = [(x * fx, y * fy) for (x, y) in sd_pts_points]

    # --- 2) Zentriert überlagern: nur Translation, sodass Bildzentren übereinstimmen ---
    h_sd, w_sd = resized_sd.shape[:2]
    h_mp, w_mp = mp_img.shape[:2]
    cx_sd, cy_sd = w_sd / 2.0, h_sd / 2.0
    cx_mp, cy_mp = w_mp / 2.0, h_mp / 2.0

    # MP -> SD (zentrierte) Koordinaten: p_sd = p_mp - c_mp + c_sd
    def mp_to_sd(p: Point) -> Point:
        x, y = p
        return (x - cx_mp + cx_sd, y - cy_mp + cy_sd)

    # Inverse Transformation: SD -> MP
    def sd_to_mp(p: Point) -> Point:
        x, y = p
        return (x - cx_sd + cx_mp, y - cy_mp + cy_mp)  # (typo check below)

    # Korrigierte inverse Transformation (oben war ein Tippfehler):
    def sd_to_mp(p: Point) -> Point:
        x, y = p
        return (x - cx_sd + cx_mp, y - cy_sd + cy_mp)

    # Ungeordnete MP-Punkte in SD-Koordinaten
    mp_to_sd_unordered: PointList = [mp_to_sd(p) for p in mp_pts_points_unordered]

    # --- 3) Sortieren (existierende Funktion nutzen) ---
    mp_to_sd_ordered: PointList = order_points_by_min_distance(sd_pts_scaled, mp_to_sd_unordered)

    # --- 4) Zurück nach MP-Koordinaten ---
    sd_to_mp_ordered: PointList = [sd_to_mp(p) for p in mp_to_sd_ordered]

    # --- 5) Ergebnis zurückgeben ---
    return sd_to_mp_ordered

Point = Tuple[float, float]

def _homography_or_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Erst Homographie, sonst Affine->Homographie, sonst None (aber wir brechen nie außen ab)."""
    H, _ = cv2.findHomography(src, dst, method=0)  # keine RANSAC/SCHWELLEN
    if H is not None and np.all(np.isfinite(H)):
        return H
    A, _inliers = cv2.estimateAffine2D(src, dst)  # 2x3
    if A is not None and np.all(np.isfinite(A)):
        H_aff = np.eye(3, dtype=np.float64)
        H_aff[:2, :3] = A
        return H_aff
    return None

def create_homography_matrix(src_points, dst_points):
    """
    src_points, dst_points: Liste von (x, y), gleiche Länge, >=4
    Nutzt ALLE Punkte (method=0), kein RANSAC.
    """
    S = np.asarray(src_points, dtype=np.float64)  # (N,2)
    D = np.asarray(dst_points, dtype=np.float64)  # (N,2)
    H, _ = cv2.findHomography(S, D, method=0)
    return H

def fmt_mat(H):
    return "\n".join("  " + "  ".join(f"{v: .6f}" for v in row) for row in H)

# Transform coordinats:
def transform_coordinates(points, H):
    """
    points: Liste [(x, y), ...] in Bildkoordinaten
    H: 3x3-Homographiematrix
    -> Liste [(x', y'), ...] in Zielkoordinaten
    """
    if H is None or len(points) == 0:
        return []
    P = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)   # (N,1,2)
    out = cv2.perspectiveTransform(P, H).reshape(-1, 2)          # (N,2)
    return [(float(x), float(y)) for x, y in out]

def compose(H2, H1):
    """Ergibt die Verkettung erst H1, dann H2 (also H2 @ H1)."""
    return None if H1 is None or H2 is None else (H2 @ H1)

# Alle Funktionen,
