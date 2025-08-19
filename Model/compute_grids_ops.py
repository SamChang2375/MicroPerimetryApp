# Model/compute_grids.py
from __future__ import annotations
import math
import numpy as np
import cv2
from typing import List, Tuple, Optional, Iterable

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

# All helper functions for testing overlap of High Res and SD, Dilating, Setting Test points and writing into XML
def _as_list(out):
    if out is None: return []
    return out if isinstance(out, (list, tuple)) else [out]

def ordered_outlines(coords_xy, W=1024, H=1024, min_len=20):
    canvas = np.zeros((H, W), np.uint8)
    pts = np.round(np.asarray(coords_xy)).astype(int)
    inb = (pts[:,0] >= 0) & (pts[:,0] < W) & (pts[:,1] >= 0) & (pts[:,1] < H)
    pts = pts[inb]
    if pts.size == 0:
        return []
    canvas[pts[:,1], pts[:,0]] = 255
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    ret = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = ret[0] if len(ret) == 2 else ret[1]
    return [c[:,0,:] for c in cnts if len(c) >= min_len]  # (M,2) in (x,y)

def line_mask_from_outlines(outlines, W=1024, H=1024, thickness=1):
    m = np.zeros((H, W), np.uint8)
    for c in _as_list(outlines):
        c = np.round(np.asarray(c)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [c], isClosed=True, color=255, thickness=thickness, lineType=cv2.LINE_8)
    return m

def filled_mask_from_outlines(outlines, W=1024, H=1024):
    m = np.zeros((H, W), np.uint8)
    polys = []
    for c in _as_list(outlines):
        c = np.round(np.asarray(c)).astype(np.int32)
        if c.ndim == 2 and c.shape[1] == 2 and len(c) >= 3:
            polys.append(c.reshape(-1, 1, 2))
    if polys:
        cv2.fillPoly(m, polys, 255)
    return m

def resample_closed_curve_uniform(curve_xy, target_step=12.0, min_spacing=8.0, max_spacing=16.0):
    P = np.asarray(curve_xy, dtype=float).reshape(-1, 2)
    if len(P) < 2:
        return P.copy()
    if np.linalg.norm(P[0] - P[-1]) > 1e-6:
        P = np.vstack([P, P[0]])

    seg = np.diff(P, axis=0)
    seglen = np.hypot(seg[:,0], seg[:,1])
    L = float(seglen.sum())
    if L <= 1e-6:
        return P[:1].copy()

    n0    = max(3, int(round(L / target_step)))
    n_min = max(3, int(np.ceil (L / max_spacing)))
    n_max = max(3, int(np.floor(L / min_spacing)))
    n = n_min if n_min > n_max else int(np.clip(n0, n_min, n_max))

    d = np.linspace(0.0, L, n, endpoint=False)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    idx = np.searchsorted(s, d, side='right') - 1
    idx = np.clip(idx, 0, len(seglen)-1)
    t = (d - s[idx]) / np.maximum(seglen[idx], 1e-12)
    Q = P[idx] + (P[idx+1] - P[idx]) * t[:, None]
    return Q

def inner_points_on_grid(sd_out, step=12, min_diam=15, W=1024, H=1024):
    sd_fill = filled_mask_from_outlines(sd_out, W, H)
    inside = (sd_fill > 0).astype(np.uint8)
    dt_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)  # float32
    r_max = float(dt_in.max())
    diam_max = 2.0 * r_max
    if diam_max < min_diam:
        return np.empty((0, 2), float), diam_max, (None, None)
    r_safe = min_diam / 2.0
    cy, cx = np.unravel_index(np.argmax(dt_in), dt_in.shape)
    x0 = cx % step; y0 = cy % step
    xs = np.arange(x0, W, step, dtype=int)
    ys = np.arange(y0, H, step, dtype=int)
    Xg, Yg = np.meshgrid(xs, ys)
    ok = (inside[Yg, Xg] > 0) & (dt_in[Yg, Xg] >= r_safe)
    pts = np.stack([Xg[ok], Yg[ok]], axis=1).astype(float)
    return pts, diam_max, (cx, cy)

def px_to_deg(pts_px: np.ndarray, W=1024, H=1024, fov_deg=36.0) -> np.ndarray:
    P = np.asarray(pts_px, float).reshape(-1, 2)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    sx, sy = fov_deg / (W - 1), fov_deg / (H - 1)
    x_deg = (P[:, 0] - cx) * sx
    y_deg = -(P[:, 1] - cy) * sy
    return np.column_stack([x_deg, y_deg])

def flatten_point_sets(obj_array):
    bags = []
    for arr in obj_array:
        A = np.asarray(arr, float).reshape(-1, 2)
        if A.size:
            bags.append(A)
    return np.vstack(bags) if bags else np.empty((0, 2), float)

# Checking Overlap
def check_segmentation_overlap(
    segA_coords, segB_coords,
    *, W=1024, H=1024, threshold=0.70, return_details=False
):
    """
    Prüft den Overlap-Score zwischen zwei Segmentierungen:
    Score = (Fläche(Intersection) / min(Fläche(A), Fläche(B))).
    Liefert True, wenn Score >= threshold, sonst False.

    Rückgabe:
      - bool (ok)
      - (optional) dict mit 'score', 'area_small', 'area_A', 'area_B'
    """
    A_out = ordered_outlines(segA_coords, W, H)
    B_out = ordered_outlines(segB_coords, W, H)
    if not A_out or not B_out:
        ok = False
        det = {'score': 0.0, 'area_small': 0, 'area_A': 0, 'area_B': 0}
        return (ok, det) if return_details else ok

    A_fill = filled_mask_from_outlines(A_out, W, H)
    B_fill = filled_mask_from_outlines(B_out, W, H)

    area_A = int(cv2.countNonZero(A_fill))
    area_B = int(cv2.countNonZero(B_fill))
    if area_A == 0 or area_B == 0:
        ok = False
        det = {'score': 0.0, 'area_small': 0, 'area_A': area_A, 'area_B': area_B}
        return (ok, det) if return_details else ok

    inter = cv2.bitwise_and(A_fill, B_fill)
    area_inter = int(cv2.countNonZero(inter))
    area_small = min(area_A, area_B)

    score = area_inter / max(1, area_small)
    ok = score >= float(threshold)

    if return_details:
        return ok, {'score': score, 'area_small': area_small, 'area_A': area_A, 'area_B': area_B}
    return ok

# ---------- helpers: dünnen Kandidatenpunkte mit Mindestabstand, existierende Punkte bleiben ----------
def _poisson_thin_with_obstacles(cands, r, fixed, W, H):
    cands = np.asarray(cands, float).reshape(-1, 2)
    fixed = np.asarray(fixed, float).reshape(-1, 2) if fixed is not None and len(fixed) else np.empty((0,2), float)
    if cands.size == 0:
        return cands

    cell = max(r, 1.0)
    nx = int(np.ceil(W / cell)); ny = int(np.ceil(H / cell))
    grid = [[[] for _ in range(nx)] for __ in range(ny)]

    def _put(p):
        ix = int(p[0] // cell); iy = int(p[1] // cell)
        ix = max(0, min(nx-1, ix)); iy = max(0, min(ny-1, iy))
        grid[iy][ix].append(p)

    for q in fixed:  # Hindernisse vorab eintragen
        _put(q)

    kept = []
    r2 = r * r

    def _ok(p):
        ix = int(p[0] // cell); iy = int(p[1] // cell)
        ix = max(0, min(nx-1, ix)); iy = max(0, min(ny-1, iy))
        for jy in range(max(0, iy-1), min(ny, iy+2)):
            for jx in range(max(0, ix-1), min(nx, ix+2)):
                for q in grid[jy][jx]:
                    dx = p[0]-q[0]; dy = p[1]-q[1]
                    if dx*dx + dy*dy < r2:
                        return False
        return True

    for p in cands:
        if _ok(p):
            kept.append(p); _put(p)

    return np.asarray(kept, float).reshape(-1, 2)


# ---------- optionaler „Greedy“ Post-Fill für 1–5 Zusatzpunkte in großen Löchern ----------
def _greedy_post_fill_center(existing_pts, allowed_mask, step, max_add=5):
    existing_pts = np.asarray(existing_pts, float).reshape(-1, 2)
    H, W = allowed_mask.shape
    if H == 0 or W == 0:
        return np.empty((0,2), float)

    occ = np.ones((H, W), np.uint8) * 255
    r = max(1, int(np.floor(step / 2)))
    for (x, y) in np.round(existing_pts).astype(int):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(occ, (x, y), r, 0, -1)

    added = []
    for _ in range(max_add):
        dt = cv2.distanceTransform((occ > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        dt *= (allowed_mask > 0).astype(np.float32)  # nur erlaubte Bereiche
        iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
        if float(dt[iy, ix]) < (step - 1e-3):
            break
        added.append((float(ix), float(iy)))
        cv2.circle(occ, (int(ix), int(iy)), r, 0, -1)

    return np.asarray(added, float)


def build_mp_point_sets(
    SD_MP_Coords,
    HR_MP_Coords,
    *,
    W=1024, H=1024,
    step=12,
    fov_deg=36.0,
    border_tol_px=1.0,
    keep_outside_rings=True,
    enforce_min_dist=True,
    post_fill_max_add=3,        # 0 = aus; >0 setzt noch 1–N Punkte in große Löcher
):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # ---- Eingaben ----
    SD = np.asarray(SD_MP_Coords, float).reshape(-1, 2)
    HR = np.asarray(HR_MP_Coords, float).reshape(-1, 2)

    SD_out = ordered_outlines(SD, W, H)
    HR_out = ordered_outlines(HR, W, H)

    if not SD_out:
        empty_obj = np.array([], dtype=object)
        empty = np.empty((0, 2), float)
        return {
            "SD_out": [], "HR_out": HR_out, "levels_before": [],
            "hit_level": 0.0, "SD_point_sets": [], "Ring_point_sets": [],
            "Hit_point_sets": [], "Inward_point_sets": [],
            "inner_pts": empty, "MP_points": empty_obj,
            "all_pts_px": empty, "all_pts_deg": empty,
        }

    # ---- DT zur SD-Linie (Außen-Logik) ----
    sd_line = line_mask_from_outlines(SD_out, W, H, thickness=1)
    hr_line = line_mask_from_outlines(HR_out, W, H, thickness=1) if HR_out else np.zeros_like(sd_line)
    dt_shell = cv2.distanceTransform((sd_line == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)

    # ---- DT innerhalb der SD-Region (Innen-Logik) ----
    sd_fill = np.zeros((H, W), np.uint8)
    for c in SD_out:
        cc = np.round(np.asarray(c)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(sd_fill, [cc], 255)
    inside_mask = (sd_fill > 0).astype(np.uint8)
    if cv2.countNonZero(inside_mask) == 0:
        dt_inside = np.zeros_like(sd_fill, np.float32)
        max_inside = 0.0
    else:
        dt_inside = cv2.distanceTransform(inside_mask, cv2.DIST_L2, 5).astype(np.float32)
        max_inside = float(dt_inside.max())

    # ---- Raster für contour() + Helfer ----
    xs = np.arange(W); ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)

    def _isoline_segments(dt_img, levels):
        # nutzt Matplotlib contour, zeigt aber nichts an
        fig, ax = plt.subplots()
        try:
            cs = ax.contour(X, Y, dt_img, levels=levels, colors='none')
        finally:
            plt.close(fig)
        out = []
        for li, lvl in enumerate(cs.levels):
            for seg in cs.allsegs[li]:
                if seg.shape[0] < 3:
                    continue
                # Alles verwerfen, das den Bildrand berührt (verhindert „Rahmen“)
                if (
                    np.any(seg[:, 0] <= 0 + border_tol_px) or
                    np.any(seg[:, 0] >= (W - 1) - border_tol_px) or
                    np.any(seg[:, 1] <= 0 + border_tol_px) or
                    np.any(seg[:, 1] >= (H - 1) - border_tol_px)
                ):
                    continue
                out.append((lvl, seg.astype(float)))
        return out

    # ---- SD-Outline-Punkte (fix) ----
    SD_point_sets = [
        resample_closed_curve_uniform(
            np.asarray(c), target_step=step,
            min_spacing=step*2/3, max_spacing=step*4/3
        )
        for c in SD_out
    ]
    sd_outline_pts = np.vstack(SD_point_sets) if SD_point_sets else np.empty((0, 2), float)

    # ---- Hit-Level & Levels davor (für Außenringe) ----
    levels_before = []
    hit_level = step  # fallback
    if keep_outside_rings and hr_line.any():
        max_k = int(np.ceil(float(dt_shell.max()) / step)) + 2
        k_hit = None
        for k in range(1, max_k + 1):
            radius = k * step
            band = (dt_shell < radius).astype(np.uint8) * 255
            uncovered = cv2.countNonZero(cv2.bitwise_and(hr_line, cv2.bitwise_not(band)))
            if uncovered == 0:
                k_hit = k
                break
            levels_before.append(radius)
        if k_hit is None:
            k_hit = max_k
        hit_level = k_hit * step

    # ---- Außenringe (fix, kein Thinning) ----
    Ring_point_sets = []
    Hit_point_sets = []
    if keep_outside_rings:
        if levels_before:
            for lvl, seg in _isoline_segments(dt_shell, levels_before):
                pts = resample_closed_curve_uniform(seg, step, step*2/3, step*4/3)
                if pts.size:
                    Ring_point_sets.append((lvl, pts))
        for lvl, seg in _isoline_segments(dt_shell, [hit_level]):
            pts = resample_closed_curve_uniform(seg, step, step*2/3, step*4/3)
            if pts.size:
                Hit_point_sets.append((lvl, pts))

    outer_sets = [p for _, p in sorted(Ring_point_sets, key=lambda t: t[0])] \
               + [p for _, p in Hit_point_sets]
    outer_pts = np.vstack(outer_sets) if outer_sets else np.empty((0, 2), float)

    # ---- Innenringe (alle 12 px nach innen), danach nur innen ausdünnen ----
    inward_levels = np.arange(step, max_inside + step*0.5, step).tolist() if max_inside > step else []
    Inward_point_sets = []
    inner_cands = np.empty((0, 2), float)
    if inward_levels:
        for lvl, seg in _isoline_segments(dt_inside, inward_levels):
            pts = resample_closed_curve_uniform(seg, step, step*2/3, step*4/3)
            if pts.size:
                Inward_point_sets.append((lvl, pts))
        if Inward_point_sets:
            inner_cands = np.vstack([p for _, p in Inward_point_sets])

    # Hindernisse: SD-Outline + Außenringe (bleiben fix)
    fixed_pts = np.vstack([sd_outline_pts, outer_pts]) if (sd_outline_pts.size or outer_pts.size) else np.empty((0, 2), float)
    if enforce_min_dist and inner_cands.size:
        inner_pts = _poisson_thin_with_obstacles(inner_cands, r=step, fixed=fixed_pts, W=W, H=H)
        if inner_pts.size == 0:
            inner_pts = np.empty((0, 2), float)
    else:
        inner_pts = inner_cands

    # ---- optional: große Löcher in der Mitte mit 1–N Zusatzpunkten schließen ----
    # ---- optional: große Löcher in der Mitte schließen ----
    if post_fill_max_add > 0:
        # Statt Erosion: nimm direkt inside_mask, damit die Mitte nicht wegschmilzt
        allowed_mask = inside_mask * 255
        current = np.vstack([fixed_pts, inner_pts]) if inner_pts.size else fixed_pts
        extra = _post_fill_with_dt(current, allowed_mask, step=step,
                                   max_add=post_fill_max_add, r_fill_ratio=0.7)
        if extra.size:
            if inner_pts.size == 0:
                inner_pts = extra
            else:
                inner_pts = np.vstack([inner_pts, extra])


    # ---- alles zusammensetzen ----
    MP_points = []
    MP_points.extend([np.asarray(A, float).reshape(-1, 2) for A in SD_point_sets])
    MP_points.extend([np.asarray(A, float).reshape(-1, 2) for A in outer_sets])
    if inner_pts.size:
        MP_points.append(np.asarray(inner_pts, float).reshape(-1, 2))
    MP_points = np.array(MP_points, dtype=object) if MP_points else np.array([], dtype=object)

    all_pts_px  = flatten_point_sets(MP_points)
    all_pts_deg = px_to_deg(all_pts_px, W=W, H=H, fov_deg=fov_deg)

    return {
        "SD_out": SD_out,
        "HR_out": HR_out,
        "levels_before": levels_before,
        "hit_level": hit_level,
        "SD_point_sets": SD_point_sets,         # [pts, ...]
        "Ring_point_sets": Ring_point_sets,     # [(level, pts), ...]
        "Hit_point_sets": Hit_point_sets,       # [(hit_level, pts), ...]
        "Inward_point_sets": Inward_point_sets, # [(level, pts), ...]
        "inner_pts": inner_pts,                 # (M,2)
        "MP_points": MP_points,                 # Objekt-Array von (Ni,2)
        "all_pts_px": all_pts_px,               # (N,2)
        "all_pts_deg": all_pts_deg,             # (N,2)
    }


def _bridge_gaps_between_points(existing_pts, allowed_mask, *,
                                gap_min=16.0, gap_max=24.0,
                                min_clear=8.0,             # neuer Punkt muss mind. so weit weg von allen sein
                                max_add=200,               # Sicherheitslimit
                                W=1024, H=1024):
    """
    Suche Punktpaare mit Abstand in [gap_min, gap_max] und füge den Mittelpunkt
    hinzu, wenn er innerhalb von allowed_mask liegt und mind. min_clear Abstand
    zu allen existierenden Punkten hat. Nutzt Grid-Hash für schnelle Nachbarsuche.
    """
    P = np.asarray(existing_pts, float).reshape(-1, 2)
    if P.size == 0:
        return np.empty((0, 2), float)

    # Grid-Hash
    cell = max(min_clear, 1.0)
    nx = int(np.ceil(W / cell)); ny = int(np.ceil(H / cell))
    grid = [[[] for _ in range(nx)] for __ in range(ny)]

    def put(p):
        ix = int(p[0] // cell); iy = int(p[1] // cell)
        ix = max(0, min(nx-1, ix)); iy = max(0, min(ny-1, iy))
        grid[iy][ix].append(p)

    for q in P:
        put(q)

    def ok(p):
        ix = int(p[0] // cell); iy = int(p[1] // cell)
        ix = max(0, min(nx-1, ix)); iy = max(0, min(ny-1, iy))
        r2 = min_clear * min_clear
        for jy in range(max(0, iy-1), min(ny, iy+2)):
            for jx in range(max(0, ix-1), min(nx, ix+2)):
                for q in grid[jy][jx]:
                    dx = p[0]-q[0]; dy = p[1]-q[1]
                    if dx*dx + dy*dy < r2:
                        return False
        return True

    # brute-force über Nachbarschaften (lokal via Grid)
    added = []
    # Kandidatenpaare: nur lokale Nachbarn prüfen
    for i in range(len(P)):
        pi = P[i]
        ix = int(pi[0] // cell); iy = int(pi[1] // cell)
        neigh = []
        for jy in range(max(0, iy-2), min(ny, iy+3)):
            for jx in range(max(0, ix-2), min(nx, ix+3)):
                neigh.extend(grid[jy][jx])

        for pj in neigh:
            if pj is pi:  # Identität
                continue
            dx = pj[0]-pi[0]; dy = pj[1]-pi[1]
            d = (dx*dx + dy*dy) ** 0.5
            if gap_min <= d <= gap_max:
                mx = (pi[0]+pj[0]) * 0.5
                my = (pi[1]+pj[1]) * 0.5
                xi, yi = int(round(mx)), int(round(my))
                if 0 <= xi < W and 0 <= yi < H and allowed_mask[yi, xi] > 0:
                    mp = np.array([mx, my], float)
                    if ok(mp):
                        added.append(mp)
                        put(mp)
                        if len(added) >= max_add:
                            return np.asarray(added, float)
    return np.asarray(added, float)

def _post_fill_with_dt(current_pts, allowed_mask, *, step, max_add=3, r_fill_ratio=0.7):
    """
    Fülle bis zu max_add zusätzliche Punkte in die größten Lücken:
    - current_pts: (N,2) aktuelle Punkte (alle fix + inner_pts)
    - allowed_mask: uint8 (H,W) == 255 wo Punkte erlaubt sind (z.B. inside_mask)
    - step: nominaler Zielabstand (z.B. 12)
    - r_fill_ratio: weicher Mindestabstand in der Fill-Phase (0.6–0.8 * step)
    Return: (M,2) zusätzliche Punkte
    """
    import numpy as np, cv2

    H, W = allowed_mask.shape
    if current_pts is None or len(current_pts) == 0:
        occ = np.zeros((H, W), np.uint8)
    else:
        occ = np.zeros((H, W), np.uint8)
        pts = np.round(current_pts).astype(int)
        pts = pts[(pts[:,0]>=0)&(pts[:,0]<W)&(pts[:,1]>=0)&(pts[:,1]<H)]
        # kleine Scheibe markieren, damit DT den realen Einflussbereich besser abbildet
        rad = max(1, int(np.floor(step/3)))
        if rad <= 1:
            occ[pts[:,1], pts[:,0]] = 255
        else:
            for x, y in pts:
                cv2.circle(occ, (x, y), rad, 255, -1, lineType=cv2.LINE_8)

    # nur zulässiger Bereich
    allowed = (allowed_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(allowed) == 0:
        return np.empty((0,2), float)

    added = []
    r_fill = max(1.0, float(step) * float(r_fill_ratio))

    for _ in range(int(max_add)):
        # DT von belegten Pixeln (0 = belegt, 255 = frei)
        free = cv2.bitwise_and(255 - occ, allowed)
        if cv2.countNonZero(free) == 0:
            break
        dt = cv2.distanceTransform((free > 0).astype(np.uint8), cv2.DIST_L2, 5)

        # größten Radius finden
        yx = np.unravel_index(np.argmax(dt), dt.shape)
        cy, cx = int(yx[0]), int(yx[1])
        dmax = float(dt[cy, cx])
        if dmax < r_fill:   # kein ausreichend großes Loch mehr
            break

        added.append((cx, cy))
        # neuen Punkt als belegt markieren (Scheibe)
        rad = max(1, int(np.floor(step/3)))
        cv2.circle(occ, (cx, cy), rad, 255, -1, lineType=cv2.LINE_8)

    return np.asarray(added, float)

# --- XML Writer: Punkte in GRAD -> XML ----------------------------------------
from typing import Iterable
import numpy as np

def _escape_attr(s: str) -> str:
    """
    XML-Attribut escapen. Unterstützt '°' ODER bereits '&#176;'.
    Sorgt dafür, dass Gradzeichen als &#176; im XML steht.
    """
    s = str(s).replace('&#176;', '°')
    s = (s.replace('&', '&amp;')
           .replace('<', '&lt;')
           .replace('>', '&gt;')
           .replace('"', '&quot;')
           .replace("'", '&apos;'))
    s = s.replace('°', '&#176;')
    return s

def write_points_xml(points_deg: Iterable[Iterable[float]],
                     xml_path: str,
                     grid_name: str = '3° Circle plus center',
                     precision: int = 3) -> None:
    """
    Schreibt die MP-Stimulusliste in eine Mit-Patterns-XML.
    Erwartet 'points_deg' als (N,2)-Array/Liste in Grad (x_deg, y_deg).
    """
    pts = np.asarray(points_deg, dtype=float).reshape(-1, 2)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE MitPatterns>')
    lines.append('<patterns>')
    lines.append('<pattern_expert>')
    lines.append(f'<Grid name="{_escape_attr(grid_name)}">')

    fmt = f'{{:.{precision}f}}'
    for i, (xd, yd) in enumerate(pts, start=1):
        lines.append(
            f'  <Stimulus id="{i}" x_deg="{fmt.format(xd)}" y_deg="{fmt.format(yd)}" />'
        )

    lines.append('</Grid>')
    lines.append('</pattern_expert>')
    lines.append('</patterns>')

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
