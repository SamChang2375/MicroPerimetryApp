from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import QRect, QRectF
from pathlib import Path
from Controller.enums import MouseStatus
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QWheelEvent
import math

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

    pointAdded = pyqtSignal(float, float)

    deleteRect = pyqtSignal(float, float, float, float)

    segEditStart = pyqtSignal(float, float)
    segEditMove = pyqtSignal(float, float)
    segEditEnd = pyqtSignal(float, float)

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
        self._overlay_width = 1

        self._points: list[QPointF] = []
        self._points_color = QColor(0, 0, 255)
        self._points_radius = 2

        self._del_active: bool = False
        self._del_start_img: QPointF | None = None
        self._del_cur_img: QPointF | None = None
        # Optics of the delet-Rect
        self._rect_pen = QPen(QColor(0, 255, 0), 1, Qt.PenStyle.DashLine)
        self._rect_brush = QBrush(QColor(0, 255, 0, 40))

        self._edit_active: bool = False
        self._edit_last_img: QPointF | None = None

        # Zooming
        self._zoom: float = 1.0
        self._center: QPointF | None = None  # Bildkoordinaten des View-Zentrum

    # ------ Zoom Function ------
    def current_scale(self) -> float:
        if not self._pixmap or self.width() <= 0 or self.height() <= 0:
            return 1.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        base = min(self.width() / iw, self.height() / ih)
        return base * self._zoom

    # Praktisch für Controller: Bildschirm-Pixel -> Bild-Pixel
    def screen_to_image_dist(self, d_screen: float) -> float:
        return d_screen / self.current_scale()

    # Beim Laden/Drop: View zurücksetzen
    def _reset_view(self):
        if not self._pixmap:
            return
        self._zoom = 1.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._center = QPointF(iw / 2.0, ih / 2.0)

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
        self._points.clear()
        self._reset_view()
        self.setProperty("dragActive", False)

        self.setPixmap(QPixmap())  # <<< wichtig: internes QLabel-Pixmap leeren
        self.imageDropped.emit(path)  # <--- zurück!
        self.setText("")
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

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
        self._points.clear()
        self._reset_view()
        self.setPixmap(QPixmap())  # <<< wichtig
        self.setText("")
        self.update()
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

    # Set points
    def set_points(self, pts_img):
        self._points = [QPointF(p[0], p[1]) if not isinstance(p, QPointF) else p for p in pts_img]
        self.update()

    def clear_points(self):
        self._points.clear()
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

    def show_qimage(self, qimg: QImage):
        # Hier WIRKLICH das Bild übernehmen – aber Zoom/Center NICHT resetten,
        # sonst "springen" Slider/Reset/Zoom.
        if qimg.isNull():
            return
        self._pixmap = QPixmap.fromImage(qimg)
        if self._center is None:
            self._reset_view()  # nur beim allerersten Bild
        self.setText("")
        self.update()

    # Mouse Actions
    def mousePressEvent(self, e):
        # Roh-Log, damit wir sehen, ob Events ankommen:
        # (Achtung: kommt auch bei IDLE)
        # print("[DropArea] mousePress", e.button(), "status=", self._status)
        if self._status == MouseStatus.DRAW_PTS and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self.pointAdded.emit(img_pt.x(), img_pt.y())  # <-- neu
                e.accept()
                return

        if self._status == MouseStatus.DRAW_SEG and self._pixmap and e.button() in DRAW_BUTTONS:
            img_pt = self._widget_to_image(e.position())
            print("[DropArea] PRESS ok, img_pt=", img_pt)  # DEBUG
            if img_pt is not None:
                self._rmb_down = True
                self.segDrawStart.emit(img_pt.x(), img_pt.y())
                e.accept(); return

        if self._status == MouseStatus.DEL_STR and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            print(f"[{self.objectName()}] DEL_PRESS at widget=({e.position().x():.1f},{e.position().y():.1f})")
            img_pt = self._widget_to_image(e.position(), allow_outside=True)
            print(
                f"[{self.objectName()}]   -> img_pt={None if img_pt is None else (round(img_pt.x(), 1), round(img_pt.y(), 1))}")
            if img_pt is not None:
                self._del_active = True
                self._del_start_img = img_pt
                self._del_cur_img = img_pt
                self.update()
                e.accept()
                return

        # --- EDIT_SEG: linken Button drücken um auf Linie "einzuhaken"
        if self._status == MouseStatus.EDIT_SEG and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self._edit_active = True
                self._edit_last_img = img_pt
                self.segEditStart.emit(img_pt.x(), img_pt.y())
                e.accept()
                return

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

        if self._status == MouseStatus.DEL_STR and self._pixmap and self._del_active and (
                e.buttons() & Qt.MouseButton.LeftButton):
            img_pt = self._widget_to_image(e.position(), allow_outside=True)
            if img_pt is not None:
                self._del_cur_img = img_pt
                print(f"[{self.objectName()}] DEL_MOVE img=({img_pt.x():.1f},{img_pt.y():.1f})")  # optional
                self.update()
                e.accept()
                return

        if self._status == MouseStatus.EDIT_SEG and self._pixmap and self._edit_active and (
                e.buttons() & Qt.MouseButton.LeftButton):
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self._edit_last_img = img_pt
                self.segEditMove.emit(img_pt.x(), img_pt.y())
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

        if self._status == MouseStatus.DEL_STR and self._del_active and e.button() == Qt.MouseButton.LeftButton:
            print(f"[{self.objectName()}] DEL_RELEASE")
            self._del_active = False
            if self._del_start_img is not None and self._del_cur_img is not None:
                x1, y1 = self._del_start_img.x(), self._del_start_img.y()
                x2, y2 = self._del_cur_img.x(), self._del_cur_img.y()
                print(f"[{self.objectName()}]   EMIT deleteRect ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")
                self.deleteRect.emit(x1, y1, x2, y2)
            self._del_start_img = None
            self._del_cur_img = None
            self.update()
            e.accept()
            return

        if self._status == MouseStatus.EDIT_SEG and self._edit_active and e.button() == Qt.MouseButton.LeftButton:
            self._edit_active = False
            if self._edit_last_img is not None:
                self.segEditEnd.emit(self._edit_last_img.x(), self._edit_last_img.y())
            self._edit_last_img = None
            e.accept()
            return

        super().mouseReleaseEvent(e)

    # Mouse Wheel = Zoom-to-Mouse
    def wheelEvent(self, e: QWheelEvent):
        if not self._pixmap:
            return
        if self._center is None:
            self._reset_view()

        # 1 "tick" = 120
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return

        old_scale = self.current_scale()
        factor = 1.2 ** steps
        new_zoom = max(0.1, min(self._zoom * factor, 40.0))  # clamp

        # Bildpunkt unter Cursor vor dem Zoom:
        img_pt = self._widget_to_image(e.position())
        self._zoom = new_zoom

        # Zentrum anpassen, sodass img_pt unter Cursor bleibt:
        if img_pt is not None:
            s = self.current_scale()
            W, H = float(self.width()), float(self.height())
            xw, yw = float(e.position().x()), float(e.position().y())
            cx = img_pt.x() - (xw - W / 2.0) / s
            cy = img_pt.y() - (yw - H / 2.0) / s
            self._center = QPointF(cx, cy)
            self._clamp_center()

        self.update()

    # Paint: Bild + Overlays in Widgetkoordinaten (pixelfest)
    def paintEvent(self, e):
        if not self._pixmap:
            return
        if self._center is None:
            self._reset_view()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        s = self.current_scale()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        W, H = self.width(), self.height()

        # Bild zeichnen: Ziel-Rect aus center+scale
        x = (W / 2.0) - self._center.x() * s
        y = (H / 2.0) - self._center.y() * s
        p.drawPixmap(int(x), int(y), int(iw * s), int(ih * s), self._pixmap)

        # Overlays: immer in Widget-Pixeln
        # 1) Seg-Linie
        if len(self._seg_points) >= 2:
            pen = QPen(self._overlay_color, self._overlay_width)
            pen.setCosmetic(True)  # LINIE BLEIBT GLEICH DICK
            p.setPen(pen)
            last = None
            for pt_img in self._seg_points:
                wpt = self._image_to_widget(pt_img)
                if last is not None:
                    p.drawLine(last, wpt)
                last = wpt

        # 2) Punkte
        if self._points:
            pen = QPen(self._points_color, 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(QBrush(self._points_color))
            r = self._points_radius  # Radius in Screen-Pixeln
            for pt_img in self._points:
                wpt = self._image_to_widget(pt_img)
                p.drawEllipse(wpt, r, r)

        # 3) Delete-Rect (falls aktiv) – wie gehabt ...
        if self._del_active and self._del_start_img is not None and self._del_cur_img is not None:
            a = self._image_to_widget(self._del_start_img)
            b = self._image_to_widget(self._del_cur_img)
            # a/b sind QPointF in Widget-Koordinaten; baue normiertes QRectF
            rect = QRectF(a, b).normalized()

            pen = self._rect_pen
            pen.setCosmetic(True)  # gleichbleibende Linienstärke
            p.setPen(pen)
            p.setBrush(self._rect_brush)
            p.drawRect(rect)

        p.end()

    # Helper Functions for Zooming function
    # -------------------------------------------------
    # Mapping: Bild <-> Widget MIT center/zoom
    def _image_to_widget(self, pt_img: QPointF) -> QPointF:
        s = self.current_scale()
        W, H = float(self.width()), float(self.height())
        cx, cy = float(self._center.x()), float(self._center.y())
        xw = W / 2.0 + (pt_img.x() - cx) * s
        yw = H / 2.0 + (pt_img.y() - cy) * s
        return QPointF(xw, yw)

    def _widget_to_image(self, posf, *, allow_outside: bool = False) -> QPointF | None:
        if not self._pixmap:
            return None
        s = self.current_scale()
        W, H = float(self.width()), float(self.height())
        cx, cy = float(self._center.x()), float(self._center.y())
        xi = cx + (float(posf.x()) - W / 2.0) / s
        yi = cy + (float(posf.y()) - H / 2.0) / s
        # Begrenzen auf Bild
        if allow_outside:
            xi = max(0.0, min(xi, self._pixmap.width() - 1.0))
            yi = max(0.0, min(yi, self._pixmap.height() - 1.0))
            return QPointF(xi, yi)
        if 0.0 <= xi < self._pixmap.width() and 0.0 <= yi < self._pixmap.height():
            return QPointF(xi, yi)
        return None

    def _clamp_center(self):
        """Sorge dafür, dass beim Zoomen nicht komplett 'ins Leere' gepannt wird."""
        if not self._pixmap or self._center is None:
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        s = self.current_scale()
        half_w_img = self.width() / (2.0 * s)
        half_h_img = self.height() / (2.0 * s)

        # Falls View größer als Bild -> Mittelpunkt setzen
        if half_w_img >= iw / 2.0:
            cx_min = cx_max = iw / 2.0
        else:
            cx_min, cx_max = half_w_img, iw - half_w_img

        if half_h_img >= ih / 2.0:
            cy_min = cy_max = ih / 2.0
        else:
            cy_min, cy_max = half_h_img, ih - half_h_img

        cx = min(max(self._center.x(), cx_min), cx_max)
        cy = min(max(self._center.y(), cy_min), cy_max)
        self._center = QPointF(cx, cy)