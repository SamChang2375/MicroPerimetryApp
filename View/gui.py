from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton
from PyQt6.QtCore import Qt
from .panel import Panel
from .dropArea import ImageDropArea
from Controller.MPController import ImageController

class MicroPerimetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroPerimetryApp")
        self.setFixedSize(1920 , 1080)
        self._init_ui()
        # Controller erstellen (verdrahtet sich selbst mit der View)
        self.controller = ImageController(self)

    def _init_ui(self):
        # Window Background
        palette = self.palette()
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Build Splitter structure
        mainSplitter = QSplitter(Qt.Orientation.Horizontal)   # links/rechts
        leftSplitter = QSplitter(Qt.Orientation.Vertical)      # oben/unten

        # create three Panels --> Implemented in the Panel-Class
        self.topLeftPanel = Panel("High Res OCT Image")
        self.bottomLeftPanel = Panel("SD OCT Image")
        self.rightPanel = Panel("MicroPerimetry Image")

        # Put the panels together to the Layout
        leftSplitter.addWidget(self.topLeftPanel)
        leftSplitter.addWidget(self.bottomLeftPanel)
        mainSplitter.addWidget(leftSplitter)
        mainSplitter.addWidget(self.rightPanel)

        # Starting panel window size + resizing panels
        leftSplitter.setSizes([200, 200])
        mainSplitter.setSizes([500, 700])
        mainSplitter.setStretchFactor(0, 1)
        mainSplitter.setStretchFactor(1, 2)
        leftSplitter.setStretchFactor(0, 1)
        leftSplitter.setStretchFactor(1, 1)

        # Set Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(mainSplitter)

        # Set up the toolbars and the Image-Drop-Areas
        self._setup_toolbars()
        self._setup_drop_areas()
        self._connect_slider_signals()

    # ------- Helper function to build the toolBar Buttons -------
    def _setup_toolbars(self):
        # HighRes Tools
        self.topLeftPanel.add_toolbar_buttons({
            "Draw Seg": self._btn("Draw Seg"),
            "Draw Pts": self._btn("Draw Pts"),
            "Edit Seg": self._btn("Edit Seg"),
            "Del Str": self._btn("Del Str"),
            "Reset": self._btn("Reset")
        })
        # ← Slider hinzufügen (mit Labels)
        self.topLeftPanel.add_adjustment_sliders(
            with_labels=True,  # "Contrast" / "Brightness" anzeigen
            contrast_range=(0, 300),  # 100 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140  # ggf. anpassen, wenn es eng wird
        )

        # Unten links: SD-OCT – Filter/Overlay
        self.bottomLeftPanel.add_toolbar_buttons({
            "Draw Seg": self._btn("Draw Seg"),
            "Draw Pts": self._btn("Draw Pts"),
            "Edit Seg": self._btn("Edit Seg"),
            "Del Str": self._btn("Del Str"),
            "Reset": self._btn("Reset")
        })
        self.bottomLeftPanel.add_adjustment_sliders(
            with_labels=True,  # "Contrast" / "Brightness" anzeigen
            contrast_range=(0, 300),  # 100 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140  # ggf. anpassen, wenn es eng wird
        )

        # Rechts: MicroPerimetry – Punkte/Alignment/Analyse
        self.rightPanel.add_toolbar_buttons({
            "Draw Pts": self._btn("Draw Pts"),
            "Reset": self._btn("Reset")
        })
        self.rightPanel.add_adjustment_sliders(
            with_labels=True,  # "Contrast" / "Brightness" anzeigen
            contrast_range=(0, 300),  # 100 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140  # ggf. anpassen, wenn es eng wird
        )

    def _setup_drop_areas(self):
        # Je Panel eine eigene Instanz -> später separat ansteuerbar
        self.dropHighRes = ImageDropArea("Bild hierher ziehen …")
        self.dropSD = ImageDropArea("Bild hierher ziehen …")
        self.dropMicro = ImageDropArea("Bild hierher ziehen …")

        # Links: Dropflächen direkt als Content
        self.topLeftPanel.set_content(self.dropHighRes)
        self.bottomLeftPanel.set_content(self.dropSD)

        # Rechts: Content-Container mit DropArea (oben) + Bottom-Bar (unten, 100px)
        self.rightContent = QWidget()
        v = QVBoxLayout(self.rightContent)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Oberer Bereich: Drop-Area (expandiert)
        v.addWidget(self.dropMicro)

        # Unterer Bereich: 100px hoch, 2 Buttons
        self.rightBottomBar = QWidget()
        self.rightBottomBar.setFixedHeight(100)
        hb = QHBoxLayout(self.rightBottomBar)
        hb.setContentsMargins(12, 12, 12, 12)
        hb.setSpacing(12)

        # Buttons anlegen + referenzieren (später leicht ansteuerbar)
        self.btnRightReset = QPushButton("Reset")
        self.btnComputeGrids = QPushButton("Compute Grids")
        self.btnRightReset.setMinimumHeight(40)
        self.btnComputeGrids.setMinimumHeight(40)

        # Buttons platzieren (links) + Stretch rechts für Luft
        hb.addWidget(self.btnRightReset)
        hb.addWidget(self.btnComputeGrids)
        hb.addStretch(1)

        # Bottom-Bar ans Ende setzen
        v.addWidget(self.rightBottomBar)

        # Den kompletten Container als Content des rechten Panels setzen
        self.rightPanel.set_content(self.rightContent)

        # (Optional) zum Testen:
        self.dropHighRes.imageDropped.connect(lambda p: print("[HighRes] Dropped:", p))
        self.dropSD.imageDropped.connect(lambda p: print("[SD] Dropped:", p))
        self.dropMicro.imageDropped.connect(lambda p: print("[Micro] Dropped:", p))

    def _connect_slider_signals(self):
        """Optional: Slider-Events verbinden (nur Beispiel-Prints)."""
        # Oben links
        if self.topLeftPanel.contrastSlider:
            self.topLeftPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[HighRes] Contrast -> {v}")
            )
        if self.topLeftPanel.brightnessSlider:
            self.topLeftPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[HighRes] Brightness -> {v}")
            )

        # Unten links
        if self.bottomLeftPanel.contrastSlider:
            self.bottomLeftPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[SD] Contrast -> {v}")
            )
        if self.bottomLeftPanel.brightnessSlider:
            self.bottomLeftPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[SD] Brightness -> {v}")
            )

        # Rechts
        if self.rightPanel.contrastSlider:
            self.rightPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[Micro] Contrast -> {v}")
            )
        if self.rightPanel.brightnessSlider:
            self.rightPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[Micro] Brightness -> {v}")
            )

    @staticmethod
    def _btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        return btn