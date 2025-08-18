# Model/compute_grids.py
from __future__ import annotations
from PyQt6.QtGui import QImage
from typing import List, Tuple
import math
import itertools
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


def qimage_to_rgb_np(qimg: QImage) -> np.ndarray:
    """
    QImage -> np.ndarray (H, W, 3) in RGB.
    Fallback, falls im ImageState kein rgb_np hinterlegt ist.
    """
    if qimg.isNull():
        return None
    # Nach RGB888 wandeln (ohne Alpha)
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w = qimg.width()
    h = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
    # QImage liefert bereits RGB888
    return arr.copy()

def extract_segmentation_line(
    image_rgb: np.ndarray,
    target_rgb: tuple[int, int, int] = (0, 255, 255),  # Cyan
    tol: int = 10,
    min_pixels: int = 20
) -> list[tuple[float, float]]:
    """
    Findet die längste zusammenhängende Linie (größte Komponente) in der Maske
    für die gegebene Farbe + Toleranz. Gibt ALLE Punkte (x,y) als float-Tuples zurück,
    ohne Ausdünnung/Sortierung.

    - image_rgb: HxWx3 (RGB, dtype uint8)
    - target_rgb: Ziel-Farbe (R,G,B)
    - tol: Farbtoleranz (je Kanal)
    - min_pixels: Untergrenze; darunter wird 'keine Linie' angenommen
    """
    if image_rgb is None:
        return []

    # Farb-Schwelle bauen (clampen auf [0,255])
    lo = np.array([max(0, target_rgb[0]-tol),
                   max(0, target_rgb[1]-tol),
                   max(0, target_rgb[2]-tol)], dtype=np.uint8)
    hi = np.array([min(255, target_rgb[0]+tol),
                   min(255, target_rgb[1]+tol),
                   min(255, target_rgb[2]+tol)], dtype=np.uint8)

    # Binärmaske in RGB
    mask = cv2.inRange(image_rgb, lo, hi)  # 255 = Treffer

    # Größte zusammenhängende Komponente finden
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []

    # Label 0 ist Hintergrund. Größte Fläche >0 suchen
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(1 + np.argmax(areas))
    if stats[max_idx, cv2.CC_STAT_AREA] < min_pixels:
        return []

    # Koordinaten der größten Komponente holen (als (y,x))
    ys, xs = np.where(labels == max_idx)

    # In (x,y) umordnen und als Float-Tuples zurückgeben
    pts = [(float(x), float(y)) for (x, y) in zip(xs, ys)]
    return pts

Pt = Tuple[float, float]

def correct_segmentation(
    pts: List[Pt],
    tol: float = 1.0,
    auto_close: bool = True,
    max_step: float = 1.0,
) -> List[Pt] | None:
    """
    Vereinigt:
      - '2-Runden'-Korrektur über Duplikate (erste Wiederholung von first, letzte Wiederholung von last)
      - optionales Auto-Schließen per gerader Verbindung (interpoliert in ~max_step Pixeln)

    Rückgabe:
      - Liste geschlossener Punkte (letzter Punkt == erster Punkt innerhalb tol)
      - oder None, wenn offen bleibt und auto_close=False
    """
    if not pts or len(pts) < 2:
        return None

    def peq(a: Pt, b: Pt) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol

    def interpolate(p: Pt, q: Pt, step: float) -> List[Pt]:
        dx, dy = q[0] - p[0], q[1] - p[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return [q]
        steps = max(1, int(math.ceil(dist / step)))
        return [(p[0] + dx * (i / steps), p[1] + dy * (i / steps)) for i in range(1, steps + 1)]

    work = list(pts)

    # --- Runde 1: nach Wiederholung des Startpunkts hinten abschneiden
    first = work[0]
    cut1 = None
    for j in range(1, len(work)):
        if peq(work[j], first):
            cut1 = j
            break
    if cut1 is not None:
        work = work[: cut1 + 1]
        # --- Runde 2: rückwärts – vor Wiederholung des Endpunkts vorne abschneiden
        last = work[-1]
        cut2 = None
        for i in range(len(work) - 2, -1, -1):
            if peq(work[i], last):
                cut2 = i
                break
        if cut2 is not None:
            work = work[cut2:]

        # Absicherung: geschlossen?
        if not peq(work[-1], work[0]):
            work.append(work[0])
        return work

    # Keine Wiederholung des Startpunkts gefunden → ggf. automatisch schließen
    if not auto_close:
        return None

    bridge = interpolate(work[-1], work[0], max_step)
    work.extend(bridge)
    if not peq(work[-1], work[0]):
        work.append(work[0])
    return work


def _interpolate_straight(p: Pt, q: Pt, max_step: float) -> List[Pt]:
    """Gerade Verbindung p->q als Punktefolge (ohne p, mit q am Ende)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return [q]
    # Anzahl Schritte so wählen, dass etwa max_step Pixel Abstand
    steps = max(1, int(math.ceil(dist / max_step)))
    return [(p[0] + dx * (i / steps), p[1] + dy * (i / steps)) for i in range(1, steps + 1)]

Point = Tuple[float, float]

def ensure_pointlists_ok(a: List[Point], b: List[Point], c: List[Point], *, min_len: int = 4) -> tuple[bool, str]:
    """Prüft, ob beide Listen >= min_len und gleich lang sind."""
    if len(a) < min_len or len(b) < min_len or len(c) < min_len:
        return (False, f"Not enough points: HR={len(a)}, SD={len(b)} (min. {min_len}).")
    if len(a) != len(b) or len(a) != len(c) or len(b) != len(c):
        return (False, f"Unterschiedliche Punktanzahl: HR={len(a)} vs. SD={len(b)}.")
    return (True, "")

def _euclid(p: Point, q: Point) -> float:
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.hypot(dx, dy)

def correct_HR_SD_order(hr_pts: List[Point], sd_pts: List[Point]) -> tuple[List[Point], List[Point]]:
    """
    Ordnet die SD-Punktliste so um, dass sie 1:1 in der Reihenfolge zur HR-Liste passt.
    Greedy-1:1-Zuordnung: Für jeden HR-Punkt wird der nächste (noch nicht verwendete) SD-Punkt gewählt.
    Gibt (hr_pts_unchanged, sd_pts_reordered) zurück.
    """
    n = len(hr_pts)
    if n != len(sd_pts):
        raise ValueError(f"Längen unterschiedlich: HR={n} vs. SD={len(sd_pts)}")
    if n == 0:
        return ([], [])

    used = [False] * n
    sd_ordered: List[Point] = []

    # Für jeden HR-Punkt den nächsten noch freien SD-Punkt wählen
    for i in range(n):
        p = hr_pts[i]
        best_j = None
        best_d = float("inf")
        for j in range(n):
            if used[j]:
                continue
            d = _euclid(p, sd_pts[j])
            if d < best_d:
                best_d = d
                best_j = j
        # Fallback (sollte nicht passieren, aber safety):
        if best_j is None:
            # nimm irgendeinen freien
            best_j = next(k for k in range(n) if not used[k])
        used[best_j] = True
        sd_ordered.append(sd_pts[best_j])

    return (list(hr_pts), sd_ordered)

Point = Tuple[float, float]

def _to_np(pts: List[Point]) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("points must be Nx2")
    return a

def _sym_homography_rmse(src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> float:
    """Symmetrischer Reprojektionsfehler (hin & zurück). Keine Exceptions."""
    n = src.shape[0]
    src_h = np.c_[src, np.ones((n, 1))]
    dst_h = np.c_[dst, np.ones((n, 1))]

    # vorwärts
    proj_d = (src_h @ H.T)
    proj_d = proj_d[:, :2] / np.maximum(1e-12, proj_d[:, 2:3])

    # rückwärts
    try:
        Hinv = np.linalg.inv(H)
        proj_s = (dst_h @ Hinv.T)
        proj_s = proj_s[:, :2] / np.maximum(1e-12, proj_s[:, 2:3])
        err = np.sqrt(np.sum((proj_d - dst) ** 2, axis=1) + np.sum((proj_s - src) ** 2, axis=1))
        return float(np.mean(err))
    except np.linalg.LinAlgError:
        # Wenn H nicht invertierbar ist, nur vorwärtsfehler doppelt werten
        err = np.sqrt(np.sum((proj_d - dst) ** 2, axis=1))
        return float(2.0 * np.mean(err))

def _normalized_l2_cost(src: np.ndarray, dst: np.ndarray) -> float:
    """Fallback-Kosten: beide Mengen zentrieren & auf Einheitsvarianz skalieren, dann mittlere Distanz."""
    def norm(x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True)
        s[s < 1e-12] = 1.0
        return (x - m) / s
    src_n = norm(src)
    dst_n = norm(dst)
    return float(np.linalg.norm(src_n - dst_n, axis=1).mean())

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

def correct_SD_MP_order(
    sd_points: List[Point],
    mp_points: List[Point],
) -> Tuple[List[Point], List[Point], np.ndarray, Dict]:
    """
    Ordnet MP-Punkte so an, dass sie bestmöglich zu den SD-Punkten passen.
    - prüft ALLE Permutationen (für N<=8 praktikabel),
    - KEINE Schwellenwerte, KEIN Abbruch,
    - gibt IMMER ein Ergebnis zurück (Permutation, H, Metriken).

    Returns:
        sd_ordered: identisch zu Eingabe (nur zur Einheitlichkeit)
        mp_ordered: MP-Punkte in SD-Reihenfolge
        H_sd2mp: 3x3 (Homographie falls möglich, sonst Affine als H, sonst Identität)
        info: { 'best_perm': tuple, 'rmse': float, 'fallback_cost': float, 'method': 'homography'|'affine'|'identity' }
    """
    S = _to_np(sd_points)
    M = _to_np(mp_points)
    N = S.shape[0]

    # Falls unterschiedliche Längen: auf min(N) kürzen (immer ein Ergebnis liefern)
    if M.shape[0] != N:
        n = min(N, M.shape[0])
        S = S[:n].copy()
        M = M[:n].copy()
        N = n

    # Triviale Fälle handhaben
    if N == 0:
        return [], [], np.eye(3, dtype=np.float64), {'best_perm': (), 'rmse': 0.0, 'fallback_cost': 0.0, 'method': 'identity'}
    if N < 4:
        # Für <4 Punkte: keine eindeutige Homographie -> wähle Permutation mit kleinstem normalisierten L2
        best_cost = float('inf'); best_perm = tuple(range(N))
        for perm in itertools.permutations(range(N)):
            Mp = M[list(perm)]
            cost = _normalized_l2_cost(S, Mp)
            if cost < best_cost:
                best_cost = cost; best_perm = perm
        Mp_best = M[list(best_perm)]
        # affine (wenn möglich), sonst Identität
        H = _homography_or_affine(S, Mp_best)
        method = 'homography' if (H is not None and not np.allclose(H, np.eye(3))) else 'identity'
        if H is None: H = np.eye(3, dtype=np.float64); method = 'identity'
        return S.tolist(), Mp_best.tolist(), H, {
            'best_perm': best_perm, 'rmse': _normalized_l2_cost(S, Mp_best),
            'fallback_cost': best_cost, 'method': method
        }

    # N >= 4 -> volle Bruteforce-Suche über Permutationen
    best_perm = None
    best_rmse = float('inf')
    best_fallback = float('inf')
    best_H: Optional[np.ndarray] = None
    best_method = 'identity'

    for perm in itertools.permutations(range(N)):
        Mp = M[list(perm)]

        H = _homography_or_affine(S, Mp)
        if H is not None:
            rmse = _sym_homography_rmse(S, Mp, H)
            method = 'homography' if H[2,2] != 0 and not np.allclose(H, np.eye(3)) else 'affine'
            # (Wir wählen *immer* das Minimum; keine Schwellen/Abbrüche)
            if rmse < best_rmse:
                best_rmse = rmse
                best_fallback = _normalized_l2_cost(S, Mp)
                best_perm = perm
                best_H = H
                best_method = method
        else:
            # Fallback-Kosten verwenden (damit wir *immer* eine Metrik haben)
            cost = _normalized_l2_cost(S, Mp)
            if cost < best_fallback:
                best_fallback = cost
                best_perm = perm
                best_H = None
                best_method = 'identity'

    Mp_best = M[list(best_perm)]
    if best_H is None:
        # Letzter Fallback: Identität
        best_H = np.eye(3, dtype=np.float64)
        # RMSE sinnvoll befüllen (normierter L2 als Surrogat)
        best_rmse = _normalized_l2_cost(S, Mp_best)
        best_method = 'identity'

    return S.tolist(), Mp_best.tolist(), best_H, {
        'best_perm': best_perm,
        'rmse': float(best_rmse),
        'fallback_cost': float(best_fallback),
        'method': best_method,
    }

# Homographie Matrizen berechnen:
# Model/compute_grids.py
import numpy as np
import cv2

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

