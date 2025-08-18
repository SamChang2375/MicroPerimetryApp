import cv2
import numpy as np
from PyQt6.QtGui import QImage

def apply_contrast_brightness(qimg: QImage, contrast_percent: int, brightness_percent: int) -> QImage:
    """
    contrast_percent: 0..200  (100 = neutral, 50 = halb, 150 = 1.5x)
    brightness_percent: -100..100  (0 = neutral)
    """
    if qimg.isNull():
        return qimg

    # Immer in RGBA8888 konvertieren (einheitliches Format)
    src = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = src.width(), src.height()
    bytes_per_line = src.bytesPerLine()

    ptr = src.bits()
    ptr.setsize(bytes_per_line * h)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bytes_per_line))

    # Nutzdaten ohne Padding: w*4
    arr = arr[:, :w * 4].reshape((h, w, 4)).astype(np.float32)

    # Parameter-Mapping
    alpha = max(0.0, contrast_percent) / 150.0        # 0..2.0
    beta = float(brightness_percent) * 2.55           # -255..255

    # Nur RGB anpassen, Alpha unverändert lassen
    rgb = arr[..., :3]
    rgb = np.clip(alpha * rgb + beta, 0, 255, out=rgb)

    out = arr.astype(np.uint8)

    # Zurück nach QImage (copy(), damit der Speicher owned ist)
    qout = QImage(out.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    return qout

def _numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    # Achtung: QImage darf nicht auf flüchtigen Speicher zeigen -> copy()
    qimg = QImage(
        rgb.data, w, h, 3 * w,
        QImage.Format.Format_RGB888
    ).copy()
    return qimg

def auto_crop_bars(path: str, black_thr: int = 25, white_thr: int = 230):
    """
    Schneidet links/rechts weiße Ränder und unten schwarze Leiste weg.
    Gibt (QImage_cropped, rgb_np, (x0, y0, x1, y1)) zurück.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # schwarze Leiste unten
    black_bar_h = 0
    for y in range(h - 1, -1, -1):
        if gray[y, w - 1] <= black_thr:
            black_bar_h += 1
        else:
            break
    y_end = max(1, h - black_bar_h)  # falls keine Leiste gefunden, bleibt h

    # weiße Leiste links
    left_w = 0
    for x in range(w):
        if gray[0, x] >= white_thr:
            left_w += 1
        else:
            break
    x_start = left_w

    # weiße Leiste rechts
    right_w = 0
    for x in range(w - 1, -1, -1):
        if gray[0, x] >= white_thr:
            right_w += 1
        else:
            break
    x_end = max(x_start + 1, w - right_w)

    # zuschneiden (am Farbbild!)
    cropped_bgr = bgr[:y_end, x_start:x_end, :].copy()
    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    qimg = _numpy_rgb_to_qimage(rgb)
    bbox = (x_start, 0, x_end, y_end)
    return qimg, rgb, bbox