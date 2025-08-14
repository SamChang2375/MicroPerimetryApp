from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from pathlib import Path

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDropArea(QLabel):
    """
    Drag&Drop-Widget für Bilder + (optional) Zeichnen einer Segmentationslinie
    per rechter Maustaste (RMB), wenn Modus 'draw_seg' aktiv ist.
    """
    imageDropped = pyqtSignal(str)

    # Bildkoordinaten (float, Pixel)
    segDrawStart = pyqtSignal(float, float)
    segDrawMove  = pyqtSignal(float, float)
    segDrawEnd   = pyqtSignal(float, float)

    def __init__(self, placeholder: str = "Drag & Drop the image here"):
        super().__init__(placeholder)
        self.setObjectName("DropArea")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        # optional: für Hover-Feedback während RMB nicht nötig
        # self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._path: str | None = None

        # Zeichnen
        self._mode: str = "idle"   # 'idle' | 'draw_seg'
        self._rmb_down: bool = False
        self._seg_points: list[QPointF] = []
        self._overlay_color = QColor(255, 0, 0)
        self._overlay_width = 2

    # -------------------- Public API --------------------
    def load_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pixmap = pm
        self._path = path
        self._seg_points.clear()
        self.setText("")      # Platzhalter aus
        self.update()         # neu malen (Bild + ggf. Overlay)
        self.imageDropped.emit(path)
        return True

    def clear_image(self):
        self._pixmap = None
        self._path = None
        self._seg_points.clear()
        self.setText("Bild hierher ziehen …")
        self.update()

    def show_qimage(self, qimg: QImage):
        self._pixmap = QPixmap.fromImage(qimg)
        self.setText("")
        self.update()

    @property
    def image_path(self) -> str | None:
        return self._path

    def set_interaction_mode(self, mode: str):
        self._mode = mode  # 'idle' oder 'draw_seg'

    def set_segmentation(self, pts_img):
        """Overlay vom Controller setzen (Bildkoordinaten)."""
        out = []
        for p in pts_img:
            if isinstance(p, QPointF):
                out.append(QPointF(p.x(), p.y()))
            else:
                x, y = p
                out.append(QPointF(float(x), float(y)))
        self._seg_points = out
        self.update()

    def clear_segmentation(self):
        self._seg_points.clear()
        self.update()

    # -------------------- Drag & Drop --------------------
    def dragEnterEvent(self, event):
        if self._has_image_url(event):
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self); self.style().polish(self)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._has_image_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self)

    def dropEvent(self, event):
        url = next((u for u in event.mimeData().urls() if u.isLocalFile()), None)
        if not url:
            event.ignore(); return

        path = url.toLocalFile()
        if Path(path).suffix.lower() not in ALLOWED_EXTS:
            event.ignore(); return

        pm = QPixmap(path)
        if pm.isNull():
            event.ignore(); return

        self._pixmap = pm
        self._path = path
        self._seg_points.clear()
        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self)
        self.setText("")
        self.update()
        self.imageDropped.emit(path)
        event.acceptProposedAction()

    # -------------------- Maus / Zeichnen --------------------
    def mousePressEvent(self, e):
        if self._mode == "draw_seg" and e.button() == Qt.MouseButton.RightButton and self._pixmap:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self._rmb_down = True
                self.segDrawStart.emit(img_pt.x(), img_pt.y())
                e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._mode == "draw_seg" and self._rmb_down and self._pixmap:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self.segDrawMove.emit(img_pt.x(), img_pt.y())
                e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._mode == "draw_seg" and self._rmb_down and e.button() == Qt.MouseButton.RightButton:
            self._rmb_down = False
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self.segDrawEnd.emit(img_pt.x(), img_pt.y())
            e.accept(); return
        super().mouseReleaseEvent(e)

    # -------------------- Malen --------------------
    def paintEvent(self, e):
        # QLabel malt NICHT automatisch dein Original wenn du es selbst verwaltest.
        # Wir zeichnen daher Image (aspect-fit) + Overlay explizit.
        super().paintEvent(e)

        if not self._pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Bild zeichnen (aspect-fit, zentriert)
        rect = self._target_rect()
        painter.drawPixmap(rect, self._pixmap)

        # Overlay-Linie oben drauf
        if len(self._seg_points) >= 2:
            pen = QPen(self._overlay_color, self._overlay_width)
            painter.setPen(pen)
            last = None
            for pt_img in self._seg_points:
                pt_w = self._image_to_widget(pt_img, rect)
                if last is not None:
                    painter.drawLine(last, pt_w)
                last = pt_w

        painter.end()

    # -------------------- Helpers --------------------
    def _has_image_url(self, event) -> bool:
        md = event.mimeData()
        if not md.hasUrls():
            return False
        for u in md.urls():
            if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in ALLOWED_EXTS:
                return True
        return False

    def _target_rect(self) -> QRect:
        """Bereich, in dem die Pixmap (aspect-fit) innerhalb des Widgets liegt."""
        if not self._pixmap:
            return QRect(0, 0, self.width(), self.height())
        W, H = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw == 0 or ih == 0:
            return QRect(0, 0, W, H)
        scale = min(W / iw, H / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        x = (W - dw) // 2
        y = (H - dh) // 2
        return QRect(x, y, dw, dh)

    def _widget_to_image(self, posf) -> QPointF | None:
        """Widget- → Bildkoordinaten. None, wenn außerhalb/degeneriert."""
        if not self._pixmap:
            return None
        rect = self._target_rect()
        if rect.width() <= 0 or rect.height() <= 0:   # <- 0-Schutz
            return None
        xw, yw = float(posf.x()), float(posf.y())
        if not rect.contains(int(xw), int(yw)):
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        xi = (xw - rect.x()) * iw / rect.width()
        yi = (yw - rect.y()) * ih / rect.height()
        # clamp
        xi = max(0.0, min(xi, iw - 1.0))
        yi = max(0.0, min(yi, ih - 1.0))
        return QPointF(xi, yi)

    def _image_to_widget(self, pt_img: QPointF, rect: QRect):
        """Bild- → Widgetkoordinaten innerhalb rect."""
        if rect.width() <= 0 or rect.height() <= 0:
            return pt_img
        iw, ih = self._pixmap.width(), self._pixmap.height()
        xw = rect.x() + (pt_img.x() * rect.width() / iw)
        yw = rect.y() + (pt_img.y() * rect.height() / ih)
        return QPointF(xw, yw)
