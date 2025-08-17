from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from Model.image_ops import apply_contrast_brightness
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable, QTimer
from PyQt6.QtGui import QImage, QImageReader
from Controller.enums import MouseStatus
import math
from Model.annotations import PanelAnnotations, AnnotationsBundle
from Model.grid_compute import GridComputer, GridResult

@dataclass
class ImageState:
    path: Optional[str] = None
    original: Optional[QImage] = None
    contrast: int = 150
    brightness: int = 0
    version: int = 0
    seg_points: list[tuple[float, float]] = field(default_factory=list)
    pts_points: list[tuple[float, float]] = field(default_factory=list)

# --- Worker Infrastruktur ---
class _ResultSignal(QObject):
    finished = pyqtSignal(str, int, QImage)  # panel_id, version, image

class _ProcessTask(QRunnable):
    def __init__(self, panel_id: str, version: int, qimg: QImage, c: int, b: int, sig: _ResultSignal):
        super().__init__()
        self.panel_id = panel_id
        self.version = version
        self.qimg = qimg
        self.c = c
        self.b = b
        self.sig = sig
        self._hr_edit = None

    def run(self):
        out = apply_contrast_brightness(self.qimg, self.c, self.b)
        self.sig.finished.emit(self.panel_id, self.version, out)


# --- Controller ---
class ImageController(QObject):
    """
    Verbindet View (gui.py) mit Model/Processing.
    """
    def __init__(self, view):
        super().__init__()
        self.view = view  # Referenzen auf DropAreas/Slider nimmt der Controller aus der View

        # Eindeutige Namen (nur fürs Debuggen / Logs)
        try:
            self.view.dropHighRes.setObjectName("dropHighRes")
            self.view.dropSD.setObjectName("dropSD")
            self.view.dropMicro.setObjectName("dropMicro")
        except Exception as e:
            print("Objektnamen setzen fehlgeschlagen:", e)

        self.pool = QThreadPool.globalInstance()
        self.states: Dict[str, ImageState] = {
            "highres": ImageState(),
            "sd": ImageState(),
            "micro": ImageState(),
        }
        self.sig = _ResultSignal()
        self.sig.finished.connect(self._on_task_finished)

        # Debounce-Timer pro Panel (Slider feuern häufig)
        self._debounce: Dict[str, QTimer] = {}
        for pid in self.states.keys():
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(30)  # ms
            t.timeout.connect(lambda pid=pid: self._launch_processing(pid))
            self._debounce[pid] = t

        # im Controller.__init__
        self.edit_R_screen = 30.0  # 30 px auf dem Bildschirm
        self.edit_strength = 1.0  # wie bisher
        self.edit_sigma = self.edit_R_screen * 0.5
        self._edit_hit_radius = 8.0  # optionaler Hit-Test

        # Which panel is active?
        self.mode: Dict[str, MouseStatus] = {
            "highres": MouseStatus.IDLE,
            "sd": MouseStatus.IDLE,
            "micro": MouseStatus.IDLE,
        }
        self._edit_last_xy: Dict[str, tuple[float, float] | None] = {
            "highres": None, "sd": None, "micro": None
        }
        self._wire_view()

        self.compute_sig = _ComputeSignal()
        self.compute_sig.finished.connect(self._on_compute_finished)

    # Status and Panel Activation - helper methods
    def _set_status(self, panel_id: str, status: MouseStatus, *, cursor=True):
        print(f"[CTRL] _set_status panel={panel_id} -> {status.name}")
        self.mode[panel_id] = status
        drop = self._drop_of(panel_id)
        if drop:
            print(f"[CTRL]   drop={drop.objectName()} pixmap? {bool(getattr(drop, '_pixmap', None))}")
            drop.set_mouse_status(status)
            drop.set_draw_cursor(cursor)

    def _status_is(self, panel_id: str, status: MouseStatus) -> bool:
        return self.mode.get(panel_id, MouseStatus.IDLE) == status

    # --- Wiring ---
    def _wire_view(self):
        v = self.view

        # Drops for all 3 panels
        v.dropHighRes.imageDropped.connect(lambda p: self.load_image("highres", p))
        v.dropSD.imageDropped.connect(lambda p: self.load_image("sd", p))
        v.dropMicro.imageDropped.connect(lambda p: self.load_image("micro", p))

        # Sliders for all 3 panels
        if v.topLeftPanel.contrastSlider:
            v.topLeftPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("highres", val))
        if v.topLeftPanel.brightnessSlider:
            v.topLeftPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("highres", val))

        if v.bottomLeftPanel.contrastSlider:
            v.bottomLeftPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("sd", val))
        if v.bottomLeftPanel.brightnessSlider:
            v.bottomLeftPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("sd", val))

        if v.topRightPanel.contrastSlider:
            v.topRightPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("micro", val))
        if v.topRightPanel.brightnessSlider:
            v.topRightPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("micro", val))

        # Reset-Buttons aus den Toolbars for all 3 panels
        btn = v.topLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("highres"))

        btn = v.bottomLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("sd"))

        btn = v.topRightPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("micro"))

        # ------ FOR HIGH RS OCT ------
        if (btn := v.topLeftPanel.toolbarButtons.get("Draw Seg")):
            btn.clicked.connect(lambda: self._draw_seg_activate("highres"))
        if (btn := v.topLeftPanel.toolbarButtons.get("Draw Pts")):
            btn.clicked.connect(lambda: self._draw_pts_activate("highres"))
        if (btn := v.topLeftPanel.toolbarButtons.get("Edit Seg")):
            btn.clicked.connect(lambda: self._edit_seg_activate("highres"))
        if (btn := v.topLeftPanel.toolbarButtons.get("Del Str")):
            btn.clicked.connect(lambda: self._del_str_activate("highres"))

        v.dropHighRes.segDrawStart.connect(lambda x, y: self._seg_start("highres", x, y))
        v.dropHighRes.segDrawMove.connect(lambda x, y: self._seg_move("highres", x, y))
        v.dropHighRes.segDrawEnd.connect(lambda x, y: self._seg_end("highres", x, y))
        v.dropHighRes.pointAdded.connect(lambda x, y: self._point_added("highres", x, y))
        v.dropHighRes.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("highres", x1, y1, x2, y2))
        v.dropHighRes.segEditStart.connect(lambda x, y: self._edit_start("highres", x, y))
        v.dropHighRes.segEditMove.connect(lambda x, y: self._edit_move("highres", x, y))
        v.dropHighRes.segEditEnd.connect(lambda x, y: self._edit_end("highres", x, y))

        # ------ FOR SD OCT ------
        if (btn := v.bottomLeftPanel.toolbarButtons.get("Draw Seg")):
            btn.clicked.connect(lambda: self._draw_seg_activate("sd"))
        if (btn := v.bottomLeftPanel.toolbarButtons.get("Draw Pts")):
            btn.clicked.connect(lambda: self._draw_pts_activate("sd"))
        if (btn := v.bottomLeftPanel.toolbarButtons.get("Edit Seg")):
            btn.clicked.connect(lambda: self._edit_seg_activate("sd"))
        if (btn := v.bottomLeftPanel.toolbarButtons.get("Del Str")):
            btn.clicked.connect(lambda: self._del_str_activate("sd"))

        v.dropSD.segDrawStart.connect(lambda x, y: self._seg_start("sd", x, y))
        v.dropSD.segDrawMove.connect(lambda x, y: self._seg_move("sd", x, y))
        v.dropSD.segDrawEnd.connect(lambda x, y: self._seg_end("sd", x, y))
        v.dropSD.pointAdded.connect(lambda x, y: self._point_added("sd", x, y))
        v.dropSD.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("sd", x1, y1, x2, y2))
        v.dropSD.segEditStart.connect(lambda x, y: self._edit_start("sd", x, y))
        v.dropSD.segEditMove.connect(lambda x, y: self._edit_move("sd", x, y))
        v.dropSD.segEditEnd.connect(lambda x, y: self._edit_end("sd", x, y))

        # ------ FOR MICRO (MP) ------
        if (btn := v.topRightPanel.toolbarButtons.get("Draw Pts")):
            btn.clicked.connect(lambda: self._draw_pts_activate("micro"))
        if (btn := v.topRightPanel.toolbarButtons.get("Del Pts")):
            btn.clicked.connect(lambda: self._del_str_activate("micro"))

        v.dropMicro.pointAdded.connect(lambda x, y: self._point_added("micro", x, y))
        v.dropMicro.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("micro", x1, y1, x2, y2))

        btn = v.bottomRightPanel.toolbarButtons.get("Comp Grids AppSeg")
        if btn:
            btn.clicked.connect(self._on_compute_grids_clicked)

        # ------ FOR COMPUTE (bottom right) ------
        if (btn := v.bottomRightPanel.toolbarButtons.get("Comp Grids AppSeg")):
            btn.clicked.connect(lambda: self._on_compute_grids_clicked(mode="appseg"))
        if (btn := v.bottomRightPanel.toolbarButtons.get("Comp Grids PreSeg")):
            btn.clicked.connect(lambda: self._on_compute_grids_clicked(mode="preseg"))
        if (btn := v.bottomRightPanel.toolbarButtons.get("Reset")):
            btn.clicked.connect(self._on_compute_reset_clicked)


    # Edit Segmentation Functionality
    def _edit_seg_activate(self, panel_id: str):
        print(f"[{panel_id}] Edit Seg")
        self._set_status(panel_id, MouseStatus.EDIT_SEG)

    def _edit_start(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.EDIT_SEG): return
        idx, dist = self._nearest_seg_point_index(self.states[panel_id].seg_points, x, y)
        if idx is None: return
        drop = self._drop_of(panel_id)
        hit_r_img = self._edit_hit_radius / max(1e-6, drop.current_scale())
        if dist > hit_r_img: return
        self._edit_last_xy[panel_id] = (x, y)

    def _edit_move(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.EDIT_SEG):
            return

        last = self._edit_last_xy.get(panel_id)
        if last is None:
            return

        lx, ly = last
        dx, dy = x - lx, y - ly
        if dx == 0 and dy == 0:
            return

        st = self.states[panel_id]
        pts = st.seg_points
        if not pts:
            return

        drop = self._drop_of(panel_id)
        s = drop.current_scale()
        R_img = self.edit_R_screen / s
        sigma = max(1.0, R_img * 0.5)
        R2 = R_img * R_img
        two_sigma2 = 2.0 * sigma * sigma

        import math
        for i, (px, py) in enumerate(pts):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 > R2:
                continue
            w = math.exp(-d2 / two_sigma2) * self.edit_strength
            pts[i] = (px + w * dx, py + w * dy)

        self._laplacian_smooth(pts, iters=1, lam=0.2 * (1.0 / max(1.0, s)) ** 0.3)
        drop.set_segmentation(pts)
        self._edit_last_xy[panel_id] = (x, y)

    def _edit_end(self, panel_id: str, x: float, y: float):
        self._edit_last_xy[panel_id] = None

    def _laplacian_smooth(self, pts, iters=1, lam=0.2):
        n = len(pts)
        for _ in range(iters):
            new = pts[:]
            for i in range(1, n - 1):
                ax, ay = pts[i - 1]
                bx, by = pts[i + 1]
                cx, cy = pts[i]
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                new[i] = (cx + lam * (mx - cx), cy + lam * (my - cy))
            pts[:] = new

    # Delete functionality
    def _del_str_activate(self, panel_id: str):
        print(f"[BTN] Del Str clicked for {panel_id}")
        self._set_status(panel_id, MouseStatus.DEL_STR)

    def _delete_rect(self, panel_id: str, x1: float, y1: float, x2: float, y2: float):
        print(f"[CTRL] _delete_rect CALLED panel={panel_id} coords=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")

        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)

        st = self.states[panel_id]

        before_pts = len(st.pts_points)
        st.pts_points = [
            (x, y) for (x, y) in st.pts_points
            if not (xmin <= x <= xmax and ymin <= y <= ymax)
        ]

        seg_deleted = False
        if st.seg_points and all(xmin <= x <= xmax and ymin <= y <= ymax for (x, y) in st.seg_points):
            st.seg_points.clear()
            seg_deleted = True

        drop = self._drop_of(panel_id)
        if drop:
            drop.set_points(st.pts_points)
            drop.set_segmentation(st.seg_points)

        print(f"[{panel_id}] DeleteRect ({xmin:.1f},{ymin:.1f})-({xmax:.1f},{ymax:.1f}) -> "
              f"removed {before_pts - len(st.pts_points)} pts{' + seg' if seg_deleted else ''}")

    # Draw Points Functions
    def _draw_pts_activate(self, panel_id: str):
        print(f"[{panel_id}] Draw Pts")
        self._set_status(panel_id, MouseStatus.DRAW_PTS)

    # --- Point-Handler (nur HighRes; analog für andere Panels möglich) ---
    def _point_added(self, panel_id: str, x: float, y: float):
        st = self.states[panel_id]
        st.pts_points.append((x, y))
        self._drop_of(panel_id).set_points(st.pts_points)

    # Draw Segmentation Functions
    def _draw_seg_activate(self, panel_id: str):
        print(f"[{panel_id}] Draw Seg")
        self.states[panel_id].seg_points.clear()
        self._set_status(panel_id, MouseStatus.DRAW_SEG)

    def _seg_start(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points = [(x, y)]
        self._drop_of(panel_id).set_segmentation(st.seg_points)

    def _seg_move(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points.append((x, y))
        self._drop_of(panel_id).set_segmentation(st.seg_points)

    def _seg_end(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points.append((x, y))
        self._drop_of(panel_id).set_segmentation(st.seg_points)

    # Helper Functions
    # --- Reset: Status sauber zurücksetzen, panel-spezifisch ---

    def reset_adjustments(self, panel_id: str):
        st = self.states[panel_id]
        self._debounce[panel_id].stop()
        st.contrast = 150
        st.brightness = 0
        st.version += 1

        panel = self._panel_of(panel_id)
        if panel and panel.contrastSlider:
            panel.contrastSlider.blockSignals(True)
            panel.contrastSlider.setValue(150)
            panel.contrastSlider.blockSignals(False)
        if panel and panel.brightnessSlider:
            panel.brightnessSlider.blockSignals(True)
            panel.brightnessSlider.setValue(0)
            panel.brightnessSlider.blockSignals(False)

        if st.original is not None:
            drop = self._drop_of(panel_id)
            if drop and hasattr(drop, "show_qimage"):
                drop.show_qimage(st.original)

        self._set_status(panel_id, MouseStatus.IDLE, cursor=False)

    # --- Public API (View ruft Controller) ---
    def load_image(self, panel_id: str, path: str):
        # QImageReader mit AutoTransform (EXIF-Orientierung beachten)
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            print(f"[Controller] Konnte Bild nicht laden: {path}")
            return

        st = self.states[panel_id]
        st.path = path
        st.original = img
        # Nach Bildwechsel direkt neu rendern
        self._schedule(panel_id)

    def on_contrast(self, panel_id: str, value: int):
        st = self.states[panel_id]
        st.contrast = value
        self._schedule(panel_id)

    def on_brightness(self, panel_id: str, value: int):
        st = self.states[panel_id]
        st.brightness = value
        self._schedule(panel_id)

    # --- Interna ---
    def _schedule(self, panel_id: str):
        # Debounce: starte Verarbeitung erst, wenn 60ms Ruhe sind
        self._debounce[panel_id].start()

    def _launch_processing(self, panel_id: str):
        st = self.states[panel_id]
        if st.original is None:
            return
        # Version hochzählen (alles Vorherige wird als "alt" markiert)
        st.version += 1
        task = _ProcessTask(panel_id, st.version, st.original, st.contrast, st.brightness, self.sig)
        self.pool.start(task)

    def _on_task_finished(self, panel_id: str, version: int, qimg: QImage):
        st = self.states[panel_id]
        # Alte (stale) Ergebnisse ignorieren
        if version != st.version:
            return
        # Ergebnis anzeigen
        drop = self._drop_of(panel_id)
        if drop:
            # Methode anzeigen: entweder vorhandene helper-Methode nutzen …
            if hasattr(drop, "show_qimage"):
                drop.show_qimage(qimg)
            else:
                # … oder direkt als Pixmap setzen
                from PyQt6.QtGui import QPixmap
                drop.setPixmap(QPixmap.fromImage(qimg))
                drop.setText("")

    def _panel_of(self, panel_id: str):
        v = self.view
        return {"highres": v.topLeftPanel, "sd": v.bottomLeftPanel, "micro": v.topRightPanel}.get(panel_id)

    def _drop_of(self, panel_id: str):
        v = self.view
        return {"highres": v.dropHighRes, "sd": v.dropSD, "micro": v.dropMicro}.get(panel_id)

    def _on_compute_grids_clicked(self, mode: str):
        # 1) Einsammeln
        bundle = AnnotationsBundle(
            highres=self._snapshot_panel("highres"),
            sd=self._snapshot_panel("sd"),
            micro=PanelAnnotations(seg=[], points=self.states["micro"].pts_points.copy())
        )
        # (Optional) Validierung je nach mode:
        # if mode == "appseg" and (not bundle.highres.seg or not bundle.sd.seg): ...

        # 2) Task starten (Du könntest mode in GridComputer übergeben, falls nötig)
        task = _ComputeTask(bundle, self.compute_sig)
        self.pool.start(task)

    # Helper Functions
    def _euclid(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) ** 0.5

    def _nearest_seg_point_index(self, pts: list[tuple[float, float]], x: float, y: float):
        """Nächster Stützpunkt zur Cursorposition (Bildkoords)."""
        best_i = None
        best_d2 = float("inf")
        for i, (px, py) in enumerate(pts):
            d2 = (px - x) * (px - x) + (py - y) * (py - y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i, best_d2 ** 0.5

    def _build_edit_window(self, pts: list[tuple[float, float]], idx_center: int, radius_px: float):
        """
        Liefert (indices, weights) für ein Fenster von ±radius_px entlang der POLYLINE
        um idx_center; Gewicht = Cosinus-Taper (1 in der Mitte -> 0 am Rand).
        """
        n = len(pts)
        indices = {idx_center}
        # Distanzen von center entlang der Kurve
        dist_from_center = {idx_center: 0.0}

        # nach links laufen
        d = 0.0
        j = idx_center
        while j > 0:
            d += self._euclid(pts[j], pts[j - 1])
            if d > radius_px: break
            j -= 1
            indices.add(j)
            dist_from_center[j] = d

        # nach rechts laufen
        d = 0.0
        j = idx_center
        while j < n - 1:
            d += self._euclid(pts[j], pts[j + 1])
            if d > radius_px: break
            j += 1
            indices.add(j)
            dist_from_center[j] = d

        # sortiert + Gewichte (Cosine window)
        idx_list = sorted(indices)
        w_list = []
        for j in idx_list:
            s = min(dist_from_center.get(j, 0.0), radius_px)
            # cos-Taper: w(0)=1, w(radius)=0, glatt
            w = 0.5 * (1.0 + math.cos(math.pi * (s / radius_px)))
            w_list.append(w)

        return idx_list, w_list

    # For Computational matters

    def _snapshot_panel(self, panel_id: str) -> PanelAnnotations:
        st = self.states[panel_id]
        # flache Kopien reichen (Tuple sind unveränderlich)
        return PanelAnnotations(
            seg=st.seg_points.copy(),
            points=st.pts_points.copy()
        )

    def _on_compute_finished(self, res: GridResult):
        # Ergebnis entgegennehmen und an die View geben
        # (aktuell nur Debug-Ausgabe; hier könntest du unten rechts etwas rendern)
        print("[Compute] Done:", res.message, res.debug)
        # Beispiel: Statuszeile, Dialog, oder Inhalte im bottomRightPanel aktualisieren

    def _on_compute_reset_clicked(self):
        # Beispiel: Statuszeile unten rechts leeren
        if hasattr(self.view, "computeStatus"):
            self.view.computeStatus.clear()

    def _round_pts(self, pts, nd=2):
        return [(round(x, nd), round(y, nd)) for (x, y) in pts]

    def dump_all_lists(self, nd: int = 2):
        def fmt(pts):
            return [(round(x, nd), round(y, nd)) for (x, y) in pts]

        print("\n===== DEBUG: Annotation-Listen =====")
        for pid, st in self.states.items():
            seg = fmt(st.seg_points)
            pts = fmt(st.pts_points)

            print(f"[{pid}] seg_points ({len(seg)}):")
            for i, (x, y) in enumerate(seg):
                print(f"    {i:3d}: ({x:.{nd}f}, {y:.{nd}f})")

            print(f"[{pid}] pts_points ({len(pts)}):")
            for i, (x, y) in enumerate(pts):
                print(f"    {i:3d}: ({x:.{nd}f}, {y:.{nd}f})")
        print("====================================\n")

# For computation
class _ComputeSignal(QObject):
    finished = pyqtSignal(object)  # GridResult

class _ComputeTask(QRunnable):
    def __init__(self, bundle: AnnotationsBundle, sig: _ComputeSignal):
        super().__init__()
        self.bundle = bundle
        self.sig = sig

    def run(self):
        comp = GridComputer()
        res = comp.compute(self.bundle)
        self.sig.finished.emit(res)

