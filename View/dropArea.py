from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from pathlib import Path
from Controller.enums import MouseStatus

DRAW_BUTTONS = {Qt.MouseButton.LeftButton}
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDropArea(QLabel):
    """
    Drag&Drop-Widget for the images, based on QLabel
    Aspect-Fit - it uses the whole area of the content panel
    Each ImageDropArea can be accessed separately.
    """
    imageDropped = pyqtSignal(str)  # stores the path of the image file

    segDrawStart = pyqtSignal(float, float)
    segDrawMove = pyqtSignal(float, float)
    segDrawEnd = pyqtSignal(float, float)

    def __init__(self, placeholder: str = "Drag & Drop the image here"):
        # Initialize the default (start-up)
        super().__init__(placeholder)
        self.setObjectName("DropArea")
        self.setAcceptDrops(True) # activate Drag- and Drop
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)

        self._pixmap: QPixmap | None = None
        self._path: str | None = None

        self._status: MouseStatus = MouseStatus.IDLE
        self._rmb_down: bool = False
        self._seg_points: list[QPointF] = []  # aktuelle Polyline in Bildkoordinaten
        self._overlay_color = QColor(255, 0, 0)
        self._overlay_width = 10

    # ---- Drag & Drop ----
    def dragEnterEvent(self, event):
        # Dragging into the dropArea Widget
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
        self._update_pixmap()
        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self)
        self.imageDropped.emit(path)
        event.acceptProposedAction()
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    # ---- Public API ----
    def set_mouse_status(self, status: MouseStatus):
        self._status = status

    def load_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pixmap = pm
        self._path = path
        self._seg_points.clear()
        self.setText("")
        self._update_pixmap()  # QLabel bekommt die skalierte Pixmap
        self.imageDropped.emit(path)
        return True

    def clear_image(self):
        self._pixmap = None
        self._path = None
        self._seg_points.clear()
        self.setText("Bild hierher ziehen …")
        self.update()

    @property
    def image_path(self) -> str | None:
        return self._path

    def set_draw_cursor(self, on: bool):
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def set_segmentation(self, pts_img):
        self._seg_points = [QPointF(p[0], p[1]) if not isinstance(p, QPointF) else p for p in pts_img]
        self.update()

    def clear_segmentation(self):
        self._seg_points.clear()
        self.update()

    # ---- Helpers ----
    def _has_image_url(self, event) -> bool:
        md = event.mimeData()
        if not md.hasUrls():
            return False
        for u in md.urls():
            if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in ALLOWED_EXTS:
                return True
        return False

    def _update_pixmap(self):
        if not self._pixmap:
            self.clear()
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setText("")  # Platzhalter ausblenden

    def show_qimage(self, qimg: QImage):
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()
        self.setText("")

    # Mouse Actions
    def mousePressEvent(self, e):
        # Roh-Log, damit wir sehen, ob Events ankommen:
        # (Achtung: kommt auch bei IDLE)
        # print("[DropArea] mousePress", e.button(), "status=", self._status)

        if self._status == MouseStatus.DRAW_SEG and self._pixmap and e.button() in DRAW_BUTTONS:
            img_pt = self._widget_to_image(e.position())
            print("[DropArea] PRESS ok, img_pt=", img_pt)  # DEBUG
            if img_pt is not None:
                self._rmb_down = True
                self.segDrawStart.emit(img_pt.x(), img_pt.y())
                e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (self._status == MouseStatus.DRAW_SEG and self._pixmap and self._rmb_down and
                (e.buttons() & Qt.MouseButton.LeftButton or e.buttons() & Qt.MouseButton.RightButton)):
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                # DEBUG:
                # print("[DropArea] move", img_pt.x(), img_pt.y())
                self.segDrawMove.emit(img_pt.x(), img_pt.y())
                e.accept()
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._status == MouseStatus.DRAW_SEG and self._rmb_down and e.button() in DRAW_BUTTONS:
            self._rmb_down = False
            img_pt = self._widget_to_image(e.position())
            print("[DropArea] release img_pt=", img_pt)  # DEBUG
            if img_pt is not None:
                self.segDrawEnd.emit(img_pt.x(), img_pt.y())
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # Paint
    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._pixmap or len(self._seg_points) < 2:
            return
        rect = self._target_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(self._overlay_color, self._overlay_width))
        last = None
        for pt_img in self._seg_points:
            wpt = self._image_to_widget(pt_img, rect)
            if last is not None:
                p.drawLine(last, wpt)
            last = wpt
        p.end()

    # Helper Functions
    def _target_rect(self) -> QRect:
        """Bereich, in dem das Bild zentriert mit KeepAspectRatio liegt."""
        if not self._pixmap:
            return QRect(0, 0, self.width(), self.height())
        W, H = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0 or W <= 0 or H <= 0:
            return QRect(0, 0, 0, 0)
        s = min(W / iw, H / ih)
        dw, dh = int(iw * s), int(ih * s)
        return QRect((W - dw) // 2, (H - dh) // 2, dw, dh)

    def _widget_to_image(self, posf) -> QPointF | None:
        """Widget→Bild-Koordinaten (float). None, wenn außerhalb/degeneriert."""
        if not self._pixmap:
            return None
        rect = self._target_rect()
        if rect.width() <= 0 or rect.height() <= 0:
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
        """Bild→Widget-Koordinaten (float) innerhalb 'rect'."""
        if rect.width() <= 0 or rect.height() <= 0:
            return pt_img
        iw, ih = self._pixmap.width(), self._pixmap.height()
        xw = rect.x() + (pt_img.x() * rect.width() / iw)
        yw = rect.y() + (pt_img.y() * rect.height() / ih)
        return QPointF(xw, yw)