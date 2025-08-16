# main.py
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
import ctypes
from View.gui import MicroPerimetryGUI

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("de.yourorg.MicroPerimetryApp")

def main():
    app = QApplication(sys.argv)

    # Display the icon - in taskbar
    icon_path = Path(__file__).resolve().parent / "Resources" / "Icon.ico"
    icon = QIcon(str(icon_path))
    # App-Icon for all windows / dialogues of th app
    app.setWindowIcon(icon)

    # Star the GUI, use the whole window
    win = MicroPerimetryGUI()
    win.setWindowIcon(icon)
    win.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
