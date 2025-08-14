# App/controller.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from Model.image_ops import apply_contrast_brightness
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable, QTimer
from PyQt6.QtGui import QImage, QImageReader, QPixmap

# --- Model ---
@dataclass
class ImageState:
    path: Optional[str] = None
    original: Optional[QImage] = None
    contrast: int = 150
    brightness: int = 0
    version: int = 0  # erhöht sich bei jeder Anforderung; hilft, alte Ergebnisse zu verwerfen


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

        # Debounce-Timer pro Panel (Slider feuern häufig)
        self._debounce: Dict[str, QTimer] = {}
        for pid in self.states.keys():
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(30)  # ms
            t.timeout.connect(lambda pid=pid: self._launch_processing(pid))
            self._debounce[pid] = t

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

        if v.rightPanel.contrastSlider:
            v.rightPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("micro", val))
        if v.rightPanel.brightnessSlider:
            v.rightPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("micro", val))

        # Draw-Seg Buttons checkable machen + verdrahten
        for pid, panel in (("highres", v.topLeftPanel),
                           ("sd", v.bottomLeftPanel),
                           ("micro", v.rightPanel)):
            btn = panel.toolbarButtons.get("Draw Seg")
            if btn:
                btn.setCheckable(True)
                btn.toggled.connect(lambda checked, pid=pid: self.set_mode(pid, "draw_seg" if checked else "idle"))

        # RMB-Zeichnen: Punkte vom View entgegennehmen
        v.dropHighRes.segDrawStart.connect(lambda x, y: self._seg_start("highres", x, y))
        v.dropHighRes.segDrawMove.connect(lambda x, y: self._seg_move("highres", x, y))
        v.dropHighRes.segDrawEnd.connect(lambda x, y: self._seg_end("highres", x, y))

        v.dropSD.segDrawStart.connect(lambda x, y: self._seg_start("sd", x, y))
        v.dropSD.segDrawMove.connect(lambda x, y: self._seg_move("sd", x, y))
        v.dropSD.segDrawEnd.connect(lambda x, y: self._seg_end("sd", x, y))

        v.dropMicro.segDrawStart.connect(lambda x, y: self._seg_start("micro", x, y))
        v.dropMicro.segDrawMove.connect(lambda x, y: self._seg_move("micro", x, y))
        v.dropMicro.segDrawEnd.connect(lambda x, y: self._seg_end("micro", x, y))

        # Reset-Buttons aus den Toolbars:
        btn = v.topLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("highres"))

        btn = v.bottomLeftPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("sd"))

        btn = v.rightPanel.toolbarButtons.get("Reset")
        if btn: btn.clicked.connect(lambda: self.reset_adjustments("micro"))

        # --- Modi ---
        def set_mode(self, panel_id: str, mode: str):
            st = self.states[panel_id]
            st.mode = mode
            drop = self._drop_of(panel_id)
            if drop:
                drop.set_interaction_mode("draw_seg" if mode == "draw_seg" else "idle")

    # --- Seg Handlers ---
    def _seg_start(self, panel_id: str, x: float, y: float):
        if self.states[panel_id].mode != "draw_seg":
            return
        self.states[panel_id].seg_points = [(x, y)]
        self._update_seg_view(panel_id)

    def _seg_move(self, panel_id: str, x: float, y: float):
        if self.states[panel_id].mode != "draw_seg":
            return
        self.states[panel_id].seg_points.append((x, y))
        self._update_seg_view(panel_id)

    def _seg_end(self, panel_id: str, x: float, y: float):
        if self.states[panel_id].mode != "draw_seg":
            return
        self.states[panel_id].seg_points.append((x, y))
        self._update_seg_view(panel_id)
        # Modus aktiv lassen, damit man weiterzeichnen kann – oder hier auf idle setzen.

    def _update_seg_view(self, panel_id: str):
        drop = self._drop_of(panel_id)
        if drop:
            drop.set_segmentation(self.states[panel_id].seg_points)


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

    def reset_adjustments(self, panel_id: str):
        st = self.states[panel_id]
        st.contrast = 150
        st.brightness = 0
        self._schedule(panel_id)
        # Slider in der View zurücksetzen (optional)
        panel = self._panel_of(panel_id)
        if panel and panel.contrastSlider and panel.brightnessSlider:
            panel.contrastSlider.blockSignals(True)
            panel.brightnessSlider.blockSignals(True)
            panel.contrastSlider.setValue(150)
            panel.brightnessSlider.setValue(0)
            panel.contrastSlider.blockSignals(False)
            panel.brightnessSlider.blockSignals(False)

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
        return {"highres": v.topLeftPanel, "sd": v.bottomLeftPanel, "micro": v.rightPanel}.get(panel_id)

    def _drop_of(self, panel_id: str):
        v = self.view
        return {"highres": v.dropHighRes, "sd": v.dropSD, "micro": v.dropMicro}.get(panel_id)

    def _on_compute_grids_clicked(self):
        print("[Controller] Compute Grids – später implementieren")
