# zoom_view.py
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtGui import QPainter, QTransform, QWheelEvent
from PyQt6.QtCore import Qt

class ZoomView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._user_zoomed = False
        self._scale_min = 0.05
        self._scale_max = 20.0

    # API für den Dialog
    def set_user_zoomed(self, val: bool):
        self._user_zoomed = bool(val)

    def wheelEvent(self, e: QWheelEvent):
        # User zoomt -> Flag setzen
        self._user_zoomed = True
        angle = e.angleDelta().y()
        if angle == 0:
            return
        # Zoomfaktor
        factor = 1.0015 ** angle   # feinfühlig
        # aktuelle Skalierung grob schätzen (aus der Transform)
        m = self.transform()
        current = (m.m11() + m.m22()) * 0.5
        new = max(self._scale_min, min(self._scale_max, current * factor))
        factor = new / current
        self.scale(factor, factor)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Nur solange der User nicht gezoomt hat, halten wir das Bild „fit“
        if not self._user_zoomed and self.scene():
            # Versuche: PixmapItem fitten, sonst ganze Szene
            items = self.scene().items()
            pix = next((it for it in items if hasattr(it, 'pixmap')), None)
            self.resetTransform()
            if pix is not None:
                self.fitInView(pix, Qt.AspectRatioMode.KeepAspectRatio)
            else:
                self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
