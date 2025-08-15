# main.py
import sys
from PyQt6.QtWidgets import QApplication
from View.gui import MicroPerimetryGUI

def main():
    app = QApplication(sys.argv)
    window = MicroPerimetryGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()