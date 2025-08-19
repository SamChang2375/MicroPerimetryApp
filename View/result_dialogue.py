from PyQt6.QtWidgets import QDialog, QVBoxLayout, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPathItem
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtWidgets import QApplication
from View.zoom_view import ZoomView


class ResultDialog(QDialog):
    """
    Zeigt MP-Bild mit Testpunkten + HR/SD-Segmentationslinien.
    Mit Zoom & Pan (über ZoomView).
    """
    def __init__(self, parent, qimg_mp,
                 points_px,              # (N,2) ndarray oder Liste in Pixel
                 hr_seg_px, sd_seg_px):  # list[(x,y)], list[(x,y)]
        super().__init__(parent)
        self.setWindowTitle("Microperimetry Grid Computing Result")

        # Fenster strikt auf 80% setzen (nur dieser Dialog!)
        scr = QApplication.primaryScreen().availableGeometry()
        w, h = int(scr.width() * 0.8), int(scr.height() * 0.8)
        self.setFixedSize(w, h)

        # Daten
        self._img = qimg_mp
        self._points = list(points_px) if points_px is not None else []
        self._hr = list(hr_seg_px) if hr_seg_px else []
        self._sd = list(sd_seg_px) if sd_seg_px else []

        # Scene/View
        self._scene = QGraphicsScene(self)
        self._view = ZoomView(self._scene, self)
        self._view.setBackgroundBrush(QColor(0, 0, 0))  # schwarz

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)

        self._build_scene()
        QTimer.singleShot(0, self._fit_initial)

    def _fit_initial(self):
        if hasattr(self, "_pixmap_item"):
            self._view.resetTransform()
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._view.centerOn(self._pixmap_item)
            # dem View sagen: das war Aut Fit, noch kein User-Zoom
            if hasattr(self._view, "set_user_zoomed"):
                self._view.set_user_zoomed(False)

    def _build_scene(self):
        self._scene.clear()
        if self._img is None or self._img.isNull():
            return

        # 1) Hintergrundbild
        pm = QPixmap.fromImage(self._img)
        self._pixmap_item = self._scene.addPixmap(pm)
        self._pixmap_item.setZValue(0)
        iw, ih = self._img.width(), self._img.height()
        self._scene.setSceneRect(QRectF(0, 0, iw, ih))

        # Stifte/Pinsel
        pen_hr = QPen(QColor(255, 80, 80), 2)
        pen_hr.setCosmetic(True)
        pen_sd = QPen(QColor(80, 255, 80), 2)
        pen_sd.setCosmetic(True)
        pen_pt = QPen(QColor(255, 200, 0), 1)
        pen_pt.setCosmetic(True)
        brush_pt = QBrush(QColor(255, 200, 0))

        # 2) HR Polyline
        if len(self._hr) >= 2:
            path = QPainterPath(QPointF(self._hr[0][0], self._hr[0][1]))
            for x, y in self._hr[1:]:
                path.lineTo(QPointF(x, y))
            item = QGraphicsPathItem(path)
            item.setPen(pen_hr)
            item.setZValue(1)
            self._scene.addItem(item)

        # 3) SD Polyline
        if len(self._sd) >= 2:
            path = QPainterPath(QPointF(self._sd[0][0], self._sd[0][1]))
            for x, y in self._sd[1:]:
                path.lineTo(QPointF(x, y))
            item = QGraphicsPathItem(path)
            item.setPen(pen_sd)
            item.setZValue(1)
            self._scene.addItem(item)

        # 4) Punkte
        r = 3.0
        for x, y in self._points:
            ell = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            ell.setPos(QPointF(float(x), float(y)))
            ell.setPen(pen_pt)
            ell.setBrush(brush_pt)
            ell.setZValue(2)
            ell.setFlag(ell.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self._scene.addItem(ell)

        self._view.resetTransform()

    # --- Export identisch zur Anzeige ---
    def save_annotated_png(self, out_path: str):
        pix = self._view.viewport().grab()
        pix.save(out_path, "PNG")
