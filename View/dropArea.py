from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import QRectF
from pathlib import Path
from Controller.enums import MouseStatus
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QWheelEvent, QImageReader

from Model.image_ops import auto_crop_bars

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} # Allowed Image Formats

class ImageDropArea(QLabel):
    # This class creates a Drag&Drop Widget for the images.

    # The Signals (stored as class attributes) that receive the information from all user-interactions
    # with the drop area (drawing points, ...)
    imageDropped = pyqtSignal(str)  # stores the path of the image file
    # Stores the coordinates of the segmentation
    segDrawStart = pyqtSignal(float, float)
    segDrawMove = pyqtSignal(float, float)
    segDrawEnd = pyqtSignal(float, float)
    # stores the next drawn point
    pointAdded = pyqtSignal(float, float)
    # stores the coordinates of the delete rectangle
    deleteRect = pyqtSignal(float, float, float, float)
    # Stores the changed coordinates of the segmentation edit
    segEditStart = pyqtSignal(float, float)
    segEditMove = pyqtSignal(float, float)
    segEditEnd = pyqtSignal(float, float)

    def __init__(self, placeholder: str = "Drag & Drop the image here"):
        # Initialize the drop area
        super().__init__(placeholder)
        self.setObjectName("DropArea")
        self.setAcceptDrops(True) # activate Drag- and Drop
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)

        self._pixmap: QPixmap | None = None
        self._path: str | None = None

        # Set Mouse status - IDLE.
        self._status: MouseStatus = MouseStatus.IDLE
        self._draw_active: bool = False

        # Initial status of the segmentation:
        self._seg_points: list[QPointF] = [] # No segmentation set
        self._overlay_color = QColor(255, 0, 0) # Red color for visualization
        self._overlay_width = 1

        # For displaying in the MP - the second segmentation line
        self._seg2_points = []
        self._overlay2_color = QColor(0, 255, 0)  # z.B. Grün
        self._overlay2_width = 1

        # Initial status of points drawn:
        self._points: list[QPointF] = [] # No drawn points yet
        self._points_color = QColor(0, 0, 255) # blue color visualization
        self._points_radius = 2

        # Initial status of delete activity
        self._del_active: bool = False # Initially not set on delete structure status
        self._del_start_img: QPointF | None = None
        self._del_cur_img: QPointF | None = None
        # Visual style of the delet-Rect
        self._rect_pen = QPen(QColor(0, 255, 0), 1, Qt.PenStyle.DashLine)
        self._rect_brush = QBrush(QColor(0, 255, 0, 40))

        # Initial status of the edit segmentation activity
        self._edit_active: bool = False # Initially not set on edit segmentation activity
        self._edit_last_img: QPointF | None = None

        # Zooming
        self._zoom: float = 1.0 # Standard: Show at 100%
        self._center: QPointF | None = None

    # Mouse events - different actions according to set Status
    def mousePressEvent(self, e):
        # If no image is loaded yet, nothing will happen!

        # If the mouse status is on DRAW_PTS, image is existing, and left mouse was clicked:
        if self._status == MouseStatus.DRAW_PTS and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position()) # Converts widget coordinates of the click to image coordinates
            if img_pt is not None:
                self.pointAdded.emit(img_pt.x(), img_pt.y())  # fires the signal pointAdded with the img_pt so that the controller can
                # take these Coordinates and further process them
                e.accept() # Marks this event as completed and closes the method
                return

        # If the mouse status is on DRAW_SEG:
        if self._status == MouseStatus.DRAW_SEG and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self._draw_active = True # Set drawing mode activated flag to true
                self.segDrawStart.emit(img_pt.x(), img_pt.y()) # Fire the start coordinates of the segmentation
                e.accept()
                return

        # If the mouse status is on DEL_STR:
        if self._status == MouseStatus.DEL_STR and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position(), allow_outside=True) # Allows starting outside the image.
            if img_pt is not None:
                self._del_active = True # Set delete mode activated flag to true
                self._del_start_img = img_pt # Corner 1 of the delete rectangle
                self._del_cur_img = img_pt # Corner 2, is updated continually as long as the mouse is pressed and the rectangle is drawn
                self.update() # Triggers a repaint - so that the delete rectangle is visualized live
                e.accept()
                return

        # If the mouse status is on EDIT_SEG:
        if self._status == MouseStatus.EDIT_SEG and self._pixmap and e.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position()) # The coordinates where the segmentation line is "grabbed"
            if img_pt is not None:
                self._edit_active = True # Set edit segmentation mode flag to true
                self._edit_last_img = img_pt # Remember the last clicked image coordinate (starting point)
                self.segEditStart.emit(img_pt.x(), img_pt.y()) # Fire signal that the edit begins on this specific coordinate
                e.accept()
                return

        super().mousePressEvent(e) # When none of the conditions above is true: Standard behavior from class e (Mouse events)

    def mouseMoveEvent(self, e):
        if self._status == MouseStatus.DRAW_SEG and self._pixmap and self._draw_active and Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self.segDrawMove.emit(img_pt.x(), img_pt.y()) # If the position lies in the image, the live signal segDrawMove is emitted and
                # collected by the controller
                e.accept()
                return

        if self._status == MouseStatus.DEL_STR and self._pixmap and self._del_active and Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position(), allow_outside=True)
            if img_pt is not None:
                self._del_cur_img = img_pt
                self.update() # live updating position and image visualization of the delete rectangle
                e.accept()
                return

        if self._status == MouseStatus.EDIT_SEG and self._pixmap and self._edit_active and Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self._edit_last_img = img_pt
                self.segEditMove.emit(img_pt.x(), img_pt.y()) # Live edit move signal of the live coordinate of the mouse
                e.accept()
                return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._status == MouseStatus.DRAW_SEG and self._draw_active and Qt.MouseButton.LeftButton:
            self._draw_active = False # Delete draw segmentation status active flag
            img_pt = self._widget_to_image(e.position())
            if img_pt is not None:
                self.segDrawEnd.emit(img_pt.x(), img_pt.y())
            e.accept()
            return

        if self._status == MouseStatus.DEL_STR and self._del_active and e.button() == Qt.MouseButton.LeftButton:
            self._del_active = False # Deactivate delete status active flag
            if self._del_start_img is not None and self._del_cur_img is not None:
                x1, y1 = self._del_start_img.x(), self._del_start_img.y()
                x2, y2 = self._del_cur_img.x(), self._del_cur_img.y()
                self.deleteRect.emit(x1, y1, x2, y2) # fire the signal with the rectangle coordinates to the controller
            self._del_start_img = None # Reset the delete rectangle coordinates
            self._del_cur_img = None
            self.update() # Update and repaint (the rectangle vanishes)
            e.accept()
            return

        if self._status == MouseStatus.EDIT_SEG and self._edit_active and Qt.MouseButton.LeftButton:
            self._edit_active = False # Deactivat edit segmentation status active flag
            if self._edit_last_img is not None:
                self.segEditEnd.emit(self._edit_last_img.x(), self._edit_last_img.y())
            self._edit_last_img = None
            e.accept()
            return

        super().mouseReleaseEvent(e)

    # Mouse Wheel = Zoom-to-Mouse
    def wheelEvent(self, e: QWheelEvent):
        # The Zoom function does not work when no image is loaded
        if not self._pixmap:
            return
        # Initialize view: Zoom = 1, Center = center of the image
        if self._center is None:
            self._reset_view()

        # 1 "tick" with the mousewheel = 120%
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        # Exponential zoom, from 0.1 to 40-factor zoom
        old_scale = self.current_scale()
        factor = 1.2 ** steps
        new_zoom = max(0.1, min(self._zoom * factor, 40.0))
        # Get the widget coordinates of the cursor position and transform them to image coordinates.
        img_pt = self._widget_to_image(e.position())
        self._zoom = new_zoom # Set the new zoom factor

        # adjust the center so that when zooming in / out, it is zoomed to the cursor position
        if img_pt is not None:
            s = self.current_scale()
            W, H = float(self.width()), float(self.height())
            xw, yw = float(e.position().x()), float(e.position().y())

            # Widget-Coordinate = ImageCenter + (Image - Center) * Scale
            # Solve to Center so that img_pt stays  on yw, xw and set these as the new center
            cx = img_pt.x() - (xw - W / 2.0) / s
            cy = img_pt.y() - (yw - H / 2.0) / s
            self._center = QPointF(cx, cy)
            self._clamp_center() # Fix the zoom center coordinates

        self.update()

    # Paint
    def paintEvent(self, e):
        # Show the Drag&Drop-Image-Placeholder text
        if self._pixmap is None:
            super().paintEvent(e)
            return
        # First initialization: Show center + standard zoom of 100%
        if self._center is None:
            self._reset_view()

        p = QPainter(self) # Start painting into the widget area
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True) # smooth scaling

        # Zoom function:
        s = self.current_scale() # The current scale of the image
        iw, ih = self._pixmap.width(), self._pixmap.height()
        W, H = self.width(), self.height()
        x = (W / 2.0) - self._center.x() * s
        y = (H / 2.0) - self._center.y() * s
        p.drawPixmap(int(x), int(y), int(iw * s), int(ih * s), self._pixmap)

        # Overlays:
        # each overlay-coordinate is converted into widget coordinates using _image_to_widget and then drawn without further transformation.
        # s is the actual image scaling, basd on the Aspect fit.
        # The base scaling of the image is base = min(W/iw, H/ih) so that the image maximally fits the widget window.
        # Then we have our own zoom factor: self._zoom --> s = base * self._zoom.
        # Center (cx, cy) is the image point that is supposed to lie in the center of the widget.
        # That means when center changes (e.g. through zoom), then the image area covered also changes.
        # Formulas:
        #   xw = W/2 + (xi - cx) * s
        #   yw = H/2 + (yi - cy) * s
        # In this way, all overlays are "pixel-fixed":
        # No matter how the zoom is, the thickness of the segmentation line and the radius of the drawn points always stay the same.
        # This is done by taking the image segmentation coordinates, and depending on zoom and scale calculating the widget points, and painting them.
        # Hence, paintEvent uses the Forward-functionality (image to widget coordinates), while wheel-event uses backwards-functionality (see method)
        # 1) Draw the segmentation line
        if len(self._seg_points) >= 0:
            pen = QPen(self._overlay_color, self._overlay_width)
            pen.setCosmetic(True)  # no matter the zoom, the line always has the same thickness
            p.setPen(pen)
            last = None
            for pt_img in self._seg_points:
                wpt = self._image_to_widget(pt_img)
                if last is not None:
                    p.drawLine(last, wpt)
                last = wpt

        if len(self._seg2_points) >= 0:
            pen2 = QPen(self._overlay2_color, self._overlay2_width)
            pen2.setCosmetic(True)
            p.setPen(pen2)
            last2 = None
            for pt_img in self._seg2_points:
                wpt2 = self._image_to_widget(pt_img)
                if last2 is not None:
                    p.drawLine(last2, wpt2)
                last2 = wpt2

        # 2) Draw the points
        if self._points:
            pen = QPen(self._points_color, 1)
            pen.setCosmetic(True) # The points will always have the same radius
            p.setPen(pen)
            p.setBrush(QBrush(self._points_color))
            r = self._points_radius
            for pt_img in self._points:
                wpt = self._image_to_widget(pt_img)
                p.drawEllipse(wpt, r, r)

        # 3) Delete Rectangle activity
        if self._del_active and self._del_start_img is not None and self._del_cur_img is not None:
            a = self._image_to_widget(self._del_start_img)
            b = self._image_to_widget(self._del_cur_img)
            rect = QRectF(a, b).normalized()
            pen = self._rect_pen
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(self._rect_brush)
            p.drawRect(rect)

        p.end()

    # Helper functions for Zoom functionality
    def current_scale(self) -> float:
        # Returns the current zoom
        if not self._pixmap or self.width() <= 0 or self.height() <= 0:
            return 1.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        base = min(self.width() / iw, self.height() / ih)
        return base * self._zoom

    def _image_to_widget(self, pt_img: QPointF) -> QPointF:
        # Forward-coordinate transformation (see explanation in paint-event)
        s = self.current_scale()
        W, H = float(self.width()), float(self.height())
        cx, cy = float(self._center.x()), float(self._center.y())
        xw = W / 2.0 + (pt_img.x() - cx) * s
        yw = H / 2.0 + (pt_img.y() - cy) * s
        return QPointF(xw, yw)

    def _widget_to_image(self, posf, *, allow_outside: bool = False) -> QPointF | None:
        # backward-coordinate transformation (see explanation in paint-event)
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
        # Fixes the center position so that during zooming (especially zooming out) the cursor does
        # not go out of the window
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

    # Resets the view when a new image is dropped.
    def _reset_view(self):
        if not self._pixmap:
            return
        self._zoom = 1.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._center = QPointF(iw / 2.0, ih / 2.0)

    # Helper functions for Drag&Drop
    def dragEnterEvent(self, event):
        if self._has_image_url(event): # if
            event.acceptProposedAction()
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            print("[DND] ignore")
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
        if Path(path).suffix.lower() not in ALLOWED:
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

        self.setPixmap(QPixmap())
        self.imageDropped.emit(path)
        self.setText("")
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    # ---- Public API ----
    def set_mouse_status(self, status: MouseStatus):
        self._status = status

    def load_image(self, panel_id: str, path: str):
        st = self.states[panel_id]
        st.path = path

        if panel_id in ("highres", "sd"):
            qimg_cropped = None;
            rgb_np = None;
            bbox = None
            try:
                qimg_cropped, rgb_np, bbox = auto_crop_bars(path)
            except Exception as e:
                print(f"[CROP] auto_crop_bars failed: {e!r}")

            if qimg_cropped is None or qimg_cropped.isNull():
                # Fallback: normal laden
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                img = reader.read()
                if img.isNull():
                    print(f"[LOAD] Konnte Bild nicht laden: {path}")
                    return
                st.original = img
                st.crop_rect = None
                st.rgb_np = None
            else:
                # zugeschnittenes Bild übernehmen
                st.original = qimg_cropped
                st.crop_rect = bbox  # z.B. (x0, y0, x1, y1)
                st.rgb_np = rgb_np  # optional: Numpy-RGB fürs Processing

        else:
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            img = reader.read()
            if img.isNull():
                print(f"[LOAD] Konnte Bild nicht laden: {path}")
                return
            st.original = img
            st.crop_rect = None
            st.rgb_np = None

        # (optional) alte Annotations beim neuen Bild leeren
        st.seg_points.clear()
        st.pts_points.clear()

        print(f"[LOAD] {panel_id}: size={st.original.size()}, crop_rect={getattr(st, 'crop_rect', None)}")
        self._schedule(panel_id)

    def clear_image(self):
        self._pixmap = None
        self._path = None
        self._seg_points.clear()
        self.setText("Drag&Drop imag here")
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

    def _has_image_url(self, event) -> bool:
        md = event.mimeData()
        if not md.hasUrls():
            return False
        for u in md.urls():
            if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in ALLOWED:
                return True
        return False

    def show_qimage(self, qimg: QImage):
        if qimg.isNull():
            return

        # alte Größe merken (falls vorher schon ein Bild vorhanden war)
        old_size = self._pixmap.size() if self._pixmap is not None else None

        pm = QPixmap.fromImage(qimg)
        size_changed = (old_size is None) or (old_size != pm.size())

        self._pixmap = pm

        if self._center is None or size_changed:
            self._reset_view()

        self.setText("")
        self.update()

    def set_segmentation2(self, pts_img):
        from PyQt6.QtCore import QPointF
        self._seg2_points = [QPointF(p[0], p[1]) if not isinstance(p, QPointF) else p for p in pts_img]
        self.update()

    def clear_segmentation2(self):
        self._seg2_points.clear()
        self.update()
