from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush
from PyQt6.QtCore import Qt, QRectF, QPointF

class ResultDialog(QDialog):
    """
    Zeigt MP-Bild mit Testpunkten + HR/SD-Segmentationslinien.
    - statisch (kein Zoom), sauber aufs Fenster eingepasst
    - kann ein annotiertes PNG speichern, identisch zur Anzeige
    """
    def __init__(self, parent, qimg_mp: QImage,
                 points_px,              # (N,2) ndarray in Pixel
                 hr_seg_px, sd_seg_px):  # list[(x,y)], list[(x,y)]
        super().__init__(parent)
        self.setWindowTitle("Microperimetry Grid Computing Result")

        # Bildschirmgeometrie holen
        scr = QApplication.primaryScreen().availableGeometry()
        w, h = int(scr.width() * 0.8), int(scr.height() * 0.8)

        # Fenster strikt auf 80% festsetzen
        self.setFixedSize(w, h)

        self._img = qimg_mp
        self._points = points_px
        self._hr = hr_seg_px
        self._sd = sd_seg_px

        # vorbereiten: gerenderte Pixmap für Anzeige
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self._label)

        self._render_and_set_pixmap()

    def _render_and_set_pixmap(self):
        """rendert eine Pixmap (angepasst an Fensterbreite/-höhe) und setzt sie im Label"""
        if self._img is None or self._img.isNull():
            self._label.setText("No MP image")
            return

        # Bildgröße -> auf aktuelle Dialoggröße einpassen (Aspect Fit)
        target_w = max(200, self.width() - 20)
        target_h = max(200, self.height() - 20)
        pm_base = QPixmap.fromImage(self._img).scaled(
            target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # In diese Pixmap malen
        pm = QPixmap(pm_base.size())
        pm.fill(Qt.GlobalColor.black)            # Hintergrund
        p = QPainter(pm)
        p.drawPixmap(0, 0, pm_base)

        # Mapping Bild->Pixmap-Koordinaten (gleichmäßiger Scale + Offsets)
        iw, ih = self._img.width(), self._img.height()
        pw, ph = pm_base.width(), pm_base.height()
        sx = pw / iw
        sy = ph / ih
        s  = min(sx, sy)
        # pm_base ist bereits aspect-fitted links/oben in pm gezeichnet
        ox = (pm.width()  - pw) * 0.5
        oy = (pm.height() - ph) * 0.5

        def map_pt(x, y):
            return QPointF(ox + x * s, oy + y * s)

        # Farben/Stile
        pen_hr = QPen(QColor(255, 80, 80), 2);  pen_hr.setCosmetic(True)  # HR = rot
        pen_sd = QPen(QColor(80, 255, 80), 2);  pen_sd.setCosmetic(True)  # SD = grün
        pen_pt = QPen(QColor(255, 200, 0), 1);  pen_pt.setCosmetic(True)  # Punkte = gelb
        brush_pt = QBrush(QColor(255, 200, 0))

        # HR-Linie
        if self._hr and len(self._hr) >= 2:
            p.setPen(pen_hr)
            last = None
            for (x, y) in self._hr:
                wpt = map_pt(x, y)
                if last is not None:
                    p.drawLine(last, wpt)
                last = wpt

        # SD-Linie
        if self._sd and len(self._sd) >= 2:
            p.setPen(pen_sd)
            last = None
            for (x, y) in self._sd:
                wpt = map_pt(x, y)
                if last is not None:
                    p.drawLine(last, wpt)
                last = wpt

        # Punkte
        if self._points is not None and len(self._points) > 0:
            p.setPen(pen_pt)
            p.setBrush(brush_pt)
            r = 3.0  # Screenspace-Radius
            for (x, y) in self._points:
                wpt = map_pt(float(x), float(y))
                p.drawEllipse(wpt, r, r)

        p.end()
        self._label.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_and_set_pixmap()

    # --- Export der gleichen Ansicht als PNG ---
    def save_annotated_png(self, out_path: str):
        # rendere in Pixmap in Zielgröße (volle Dialoggröße)
        pm = QPixmap(self.width(), self.height())
        pm.fill(Qt.GlobalColor.black)
        # Einfach das Label neu erzeugen:
        self._render_and_set_pixmap()
        # Pixmap vom Label holen und in die Ziel-Pixmap kopieren (mittig)
        shown = self._label.pixmap()
        painter = QPainter(pm)
        if shown:
            x = (pm.width()  - shown.width())  // 2
            y = (pm.height() - shown.height()) // 2
            painter.drawPixmap(x, y, shown)
        painter.end()
        pm.save(out_path, "PNG")