# main.py
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt
from View.gui import MicroPerimetryGUI

def main():
    app = QApplication(sys.argv)
    window = MicroPerimetryGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()