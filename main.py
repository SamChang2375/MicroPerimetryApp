# main.py
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

# (nur Windows) – sorgt dafür, dass die Taskleiste dein Icon auch beim Start aus Python zeigt
if sys.platform.startswith("win"):
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("de.yourorg.MicroPerimetryApp")
    except Exception:
        pass

from View.gui import MicroPerimetryGUI

def main():
    app = QApplication(sys.argv)

    # Pfad zum Icon (liegt bei dir: ./Images/Icon.ico)
    icon_path = Path(__file__).resolve().parent / "Resources" / "Icon.ico"
    icon = QIcon(str(icon_path))

    # 1) Default-Icon für alle Fenster/Dialogs
    app.setWindowIcon(icon)

    # 2) (optional) explizit am Hauptfenster setzen
    win = MicroPerimetryGUI()
    win.setWindowIcon(icon)

    win.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
