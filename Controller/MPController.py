# App/controller.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from Model.image_ops import apply_contrast_brightness
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable, QTimer
from PyQt6.QtGui import QImage, QImageReader, QPixmap
from Controller.enums import MouseStatus
import math

# --- Model ---
@dataclass
class ImageState:
    path: Optional[str] = None  # <-- Falls das ein Tippfehler ist: sollte 'str | None' sein!
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
        self.pool = QThreadPool.globalInstance()
        self.states: Dict[str, ImageState] = {
            "highres": ImageState(),
            "sd": ImageState(),
            "micro": ImageState(),
        }
        self.sig = _ResultSignal()
        self.sig.finished.connect(self._on_task_finished)
        self.status = MouseStatus.IDLE

        # Debounce-Timer pro Panel (Slider feuern häufig)
        self._debounce: Dict[str, QTimer] = {}
        for pid in self.states.keys():
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(30)  # ms
            t.timeout.connect(lambda pid=pid: self._launch_processing(pid))
            self._debounce[pid] = t

        self._edit_anchor_idx: int | None = None
        self._edit_window_idx: list[int] = []
        self._edit_weights: list[float] = []
        self._edit_last_xy: tuple[float, float] | None = None
        # einstellbare Parameter
        self._edit_hit_radius = 8.0  # px um überhaupt „zu greifen“
        self._edit_window_radius_px = 15.0  # ±15 px entlang der Kurve => 30 px total
        self._wire_view()

    # --- Wiring ---
    def _wire_view(self):
        v = self.view

        # Drops
        v.dropHighRes.imageDropped.connect(lambda p: self.load_image("highres", p))
        v.dropSD.imageDropped.connect(lambda p: self.load_image("sd", p))
        v.dropMicro.imageDropped.connect(lambda p: self.load_image("micro", p))

        # Slider
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

        # Reset-Buttons aus den Toolbars:
        btn = v.topLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("highres"))

        btn = v.bottomLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("sd"))

        btn = v.topRightPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("micro"))

        # Draw Segmentations
        drawSegHRbtn = v.topLeftPanel.toolbarButtons.get("Draw Seg")
        drawSegHRbtn.clicked.connect(self.HR_draw_seg_activate)

        # DropArea-Events für HighRes (live Punkte)
        v.dropHighRes.segDrawStart.connect(self._hr_seg_start)
        v.dropHighRes.segDrawMove.connect(self._hr_seg_move)
        v.dropHighRes.segDrawEnd.connect(self._hr_seg_end)
        v.dropHighRes.segDrawStart.connect(lambda x, y: print(f"[Signal] start ({x:.1f}, {y:.1f})"))
        v.dropHighRes.segDrawMove.connect(lambda x, y: None)  # zu laut; ggf. testweise: print(...)
        v.dropHighRes.segDrawEnd.connect(lambda x, y: print(f"[Signal] end   ({x:.1f}, {y:.1f})"))

        # Draw Points
        drawPtsHRbtn = v.topLeftPanel.toolbarButtons.get("Draw Pts")  # <-- neu
        if drawPtsHRbtn:
            drawPtsHRbtn.clicked.connect(self.HR_draw_pts_activate)
        v.dropHighRes.pointAdded.connect(self._hr_point_added)  # <-- neu
        v.dropHighRes.pointAdded.connect(lambda x, y: print(f"[Signal] point ({x:.1f}, {y:.1f})"))  # Console-Log

        # Delete Structures
        delStrBtn = v.topLeftPanel.toolbarButtons.get("Del Str")
        if delStrBtn:
            delStrBtn.clicked.connect(self.HR_del_str_activate)
        # Rechteck-Ergebnis aus der DropArea anhören:
        v.dropHighRes.deleteRect.connect(self._hr_delete_rect)

        # in _wire_view() nach den anderen Buttons:
        editSegBtn = v.topLeftPanel.toolbarButtons.get("Edit Seg")
        if editSegBtn:
            editSegBtn.clicked.connect(self.HR_edit_seg_activate)
        # DropArea-Signale anhören:
        v.dropHighRes.segEditStart.connect(self._hr_edit_start)
        v.dropHighRes.segEditMove.connect(self._hr_edit_move)
        v.dropHighRes.segEditEnd.connect(self._hr_edit_end)

    # Edit Segmentation Functionality
    def HR_edit_seg_activate(self):
        print("Edit Seg clicked!")
        self.status = MouseStatus.EDIT_SEG
        drop = self.view.dropHighRes
        if drop:
            drop.set_mouse_status(MouseStatus.EDIT_SEG)
            drop.set_draw_cursor(True)

    import math

    def _hr_edit_start(self, x: float, y: float):
        if self.status != MouseStatus.EDIT_SEG:
            return
        st = self.states["highres"]
        pts = st.seg_points
        if not pts:
            return

        idx, d = self._nearest_seg_point_index(pts, x, y)
        if idx is None or d > self._edit_hit_radius:
            # nichts unter dem Cursor – Session ignorieren
            return

        self._edit_anchor_idx = idx
        self._edit_window_idx, self._edit_weights = self._build_edit_window(pts, idx, self._edit_window_radius_px)
        self._edit_last_xy = (x, y)
        # optional: print(f"[HighRes] EDIT start idx={idx}, window={len(self._edit_window_idx)}")

    def _hr_edit_move(self, x: float, y: float):
        if self.status != MouseStatus.EDIT_SEG or self._edit_last_xy is None or self._edit_anchor_idx is None:
            return
        st = self.states["highres"]
        pts = st.seg_points
        if not pts:
            return

        lx, ly = self._edit_last_xy
        dx, dy = (x - lx), (y - ly)
        if dx == 0 and dy == 0:
            return

        for j, w in zip(self._edit_window_idx, self._edit_weights):
            px, py = pts[j]
            pts[j] = (px + w * dx, py + w * dy)  # 2D-Verschiebung

        self._edit_last_xy = (x, y)
        self.view.dropHighRes.set_segmentation(pts)

    def _hr_edit_end(self, x: float, y: float):
        if self.status != MouseStatus.EDIT_SEG:
            return
        self._edit_anchor_idx = None
        self._edit_window_idx = []
        self._edit_weights = []
        self._edit_last_xy = None

    # Delete functionality
    def HR_del_str_activate(self):
        print("Del Str clicked!")
        self.status = MouseStatus.DEL_STR
        drop = self.view.dropHighRes
        if drop:
            drop.set_mouse_status(MouseStatus.DEL_STR)
            drop.set_draw_cursor(True)  # Crosshair ist praktisch
        # nichts löschen, vorhandene Strukturen bleiben sichtbar

    def _hr_delete_rect(self, x1: float, y1: float, x2: float, y2: float):
        # Normalisieren
        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)

        st = self.states["highres"]

        # Punkte: alle innerhalb löschen
        before_pts = len(st.pts_points)
        st.pts_points = [
            (x, y) for (x, y) in st.pts_points
            if not (xmin <= x <= xmax and ymin <= y <= ymax)
        ]

        # Segmentation: nur löschen, wenn *alle* Punkte im Rechteck liegen
        seg_deleted = False
        if st.seg_points:
            if all(xmin <= x <= xmax and ymin <= y <= ymax for (x, y) in st.seg_points):
                st.seg_points.clear()
                seg_deleted = True

        # View updaten
        drop = self.view.dropHighRes
        drop.set_points(st.pts_points)
        drop.set_segmentation(st.seg_points)

        print(f"[HighRes] DeleteRect ({xmin:.1f},{ymin:.1f})-({xmax:.1f},{ymax:.1f}) -> "
              f"removed {before_pts - len(st.pts_points)} pts{' + seg' if seg_deleted else ''}")

    # Draw Points Functions
    def HR_draw_pts_activate(self):  # <-- neu
        print("Draw Pts clicked!")
        self.status = MouseStatus.DRAW_PTS
        drop = self.view.dropHighRes
        if drop:
            drop.set_mouse_status(MouseStatus.DRAW_PTS)
            drop.set_draw_cursor(True)
        # NICHTS löschen – vorhandene Seg/Points bleiben sichtbar

    # --- Point-Handler (nur HighRes; analog für andere Panels möglich) ---
    def _hr_point_added(self, x: float, y: float):  # <-- neu
        st = self.states["highres"]
        st.pts_points.append((x, y))
        self.view.dropHighRes.set_points(st.pts_points)
        print(f"[HighRes] POINT  ({x:.1f}, {y:.1f})")

    # Draw Segmentation Functions
    def HR_draw_seg_activate(self):
        print("Draw Seg clicked!")
        self.status = MouseStatus.DRAW_SEG
        drop = self.view.dropHighRes
        if drop:
            drop.set_mouse_status(MouseStatus.DRAW_SEG)
            drop.set_draw_cursor(True)
            print("[Controller] status:", self.status)  # DEBUG
        self.states["highres"].seg_points.clear()

    def _hr_seg_start(self, x: float, y: float):
        if self.status != MouseStatus.DRAW_SEG:
            return
        print(f"[HighRes] START  ({x:.1f}, {y:.1f})")
        st = self.states["highres"]
        st.seg_points = [(x, y)]
        self.view.dropHighRes.set_segmentation(st.seg_points)

    def _hr_seg_move(self, x: float, y: float):
        if self.status != MouseStatus.DRAW_SEG:
            return
        # Achtung: sehr „chattig“. Optional drosseln (siehe unten).
        print(f"[HighRes] MOVE   ({x:.1f}, {y:.1f})")
        st = self.states["highres"]
        st.seg_points.append((x, y))
        self.view.dropHighRes.set_segmentation(st.seg_points)

    def _hr_seg_end(self, x: float, y: float):
        if self.status != MouseStatus.DRAW_SEG:
            return
        print(f"[HighRes] END    ({x:.1f}, {y:.1f})")  # <-- Log
        st = self.states["highres"]
        st.seg_points.append((x, y))
        self.view.dropHighRes.set_segmentation(st.seg_points)

    # Helper Functions
    def reset_adjustments(self, panel_id: str):
        """Slider zurück auf neutral und ORIGINALBILD anzeigen."""
        st = self.states[panel_id]
        # ausstehende Debounces stoppen, alte Tasks werden per version ignoriert
        self._debounce[panel_id].stop()

        # State neutral
        st.contrast = 150
        st.brightness = 0
        st.version += 1  # markiert alle alten Ergebnisse als veraltet

        # View-Slider zurücksetzen (ohne Events zu feuern)
        panel = self._panel_of(panel_id)
        if panel:
            if panel.contrastSlider:
                panel.contrastSlider.blockSignals(True)
                panel.contrastSlider.setValue(150)
                panel.contrastSlider.blockSignals(False)
            if panel.brightnessSlider:
                panel.brightnessSlider.blockSignals(True)
                panel.brightnessSlider.setValue(0)
                panel.brightnessSlider.blockSignals(False)

        # Original anzeigen (ohne erneut zu rechnen)
        if st.original is not None:
            drop = self._drop_of(panel_id)
            if drop:
                if hasattr(drop, "show_qimage"):
                    drop.show_qimage(st.original)  # bequemer Helper
                else:
                    drop.setPixmap(QPixmap.fromImage(st.original))
                    drop.setText("")

        if panel_id == "highres":
            self.status = MouseStatus.IDLE
            drop = self._drop_of("highres")
            if drop:
                drop.set_mouse_status(MouseStatus.IDLE)
                drop.set_draw_cursor(False)

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

    def _on_compute_grids_clicked(self):
        print("[Controller] Compute Grids – später implementieren")

    # Helper Functions
    def _euclid(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a;
        bx, by = b
        dx = ax - bx;
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

