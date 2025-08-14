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
    alpha = max(0.0, contrast_percent) / 100.0        # 0..2.0
    beta = float(brightness_percent) * 2.55           # -255..255

    # Nur RGB anpassen, Alpha unverändert lassen
    rgb = arr[..., :3]
    rgb = np.clip(alpha * rgb + beta, 0, 255, out=rgb)

    out = arr.astype(np.uint8)

    # Zurück nach QImage (copy(), damit der Speicher owned ist)
    qout = QImage(out.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    return qout