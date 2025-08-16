from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton
from PyQt6.QtCore import Qt
from .dropArea import ImageDropArea
from Controller.MPController import ImageController
from .panel import Panel

class MicroPerimetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroPerimetryApp")
        self._init_ui()
        self.controller = ImageController(self)

    def _init_ui(self):
        # Window Background
        palette = self.palette()
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setMinimumSize(900, 600)

        # Build Splitter structure
        mainSplitter = QSplitter(Qt.Orientation.Horizontal)   # links/rechts
        leftSplitter = QSplitter(Qt.Orientation.Vertical)      # oben/unten
        rightSplitter = QSplitter(Qt.Orientation.Vertical)

        # create three Panels --> Implemented in the Panel-Class
        self.topLeftPanel = Panel("High Res OCT Image")
        self.bottomLeftPanel = Panel("SD OCT Image")
        self.topRightPanel = Panel("MicroPerimetry Imabge")
        self.bottomRightPanel = Panel("Compute Area")

        # Put the panels together to the Layout
        leftSplitter.addWidget(self.topLeftPanel)
        leftSplitter.addWidget(self.bottomLeftPanel)
        rightSplitter.addWidget(self.topRightPanel)
        rightSplitter.addWidget(self.bottomRightPanel)
        mainSplitter.addWidget(leftSplitter)
        mainSplitter.addWidget(rightSplitter)

        mainSplitter.setStretchFactor(0, 1)  # links
        mainSplitter.setStretchFactor(1, 2)  # rechts

        # oben/unten
        leftSplitter.setStretchFactor(0, 1)
        leftSplitter.setStretchFactor(1, 1)

        rightSplitter.setStretchFactor(0, 3)  # MP oben
        rightSplitter.setStretchFactor(1, 1)  # Compute unten
        # die setSizes(...) kannst du dann weglassen

        # Set Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(mainSplitter)

        # Set up the toolbars and the Image-Drop-Areas
        self._setup_toolbars()
        self._setup_drop_areas()
        self._connect_slider_signals()
        self._connect_draw_seg()

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
        self.topLeftPanel.add_adjustment_sliders(
            with_labels=True,  # "Contrast" / "Brightness" anzeigen
            contrast_range=(0, 300),  # 100 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140  # ggf. anpassen, wenn es eng wird
        )
        self.topLeftPanel.toolbarButtons["Draw Seg"].setToolTip("Segmentation-Linie zeichnen (Freihand).")
        self.topLeftPanel.toolbarButtons["Draw Pts"].setToolTip("Einzelne Punkte setzen.")
        self.topLeftPanel.toolbarButtons["Edit Seg"].setToolTip("Seg-Linie lokal verschieben (Bearbeiten).")
        self.topLeftPanel.toolbarButtons["Del Str"].setToolTip("Rechteck ziehen, um Punkte/Linie zu löschen.")
        self.topLeftPanel.toolbarButtons["Reset"].setToolTip("Bild & Slider zurücksetzen.")

        # SD OCT Tools
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
        self.bottomLeftPanel.toolbarButtons["Draw Seg"].setToolTip("Segmentation-Linie im SD-Bild zeichnen.")
        self.bottomLeftPanel.toolbarButtons["Draw Pts"].setToolTip("Punkte im SD-Bild setzen.")
        self.bottomLeftPanel.toolbarButtons["Edit Seg"].setToolTip("Seg-Linie im SD-Bild bearbeiten.")
        self.bottomLeftPanel.toolbarButtons["Del Str"].setToolTip("Bereich im SD-Bild löschen.")
        self.bottomLeftPanel.toolbarButtons["Reset"].setToolTip("Zurücksetzen (SD).")

        # Microperimetry Tools
        self.topRightPanel.add_toolbar_buttons({
            "Draw Pts": self._btn("Draw Pts"),
            "Del Pts": self._btn("Del Pts"),
            "Reset": self._btn("Reset")
        })
        self.topRightPanel.add_adjustment_sliders(
            with_labels=True,  # "Contrast" / "Brightness" anzeigen
            contrast_range=(0, 300),  # 100 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140  # ggf. anpassen, wenn es eng wird
        )
        self.topRightPanel.toolbarButtons["Draw Pts"].setToolTip("MP-Punkte setzen.")
        self.topRightPanel.toolbarButtons["Del Pts"].setToolTip("Bereich ziehen, um MP-Punkte zu löschen.")
        self.topRightPanel.toolbarButtons["Reset"].setToolTip("Zurücksetzen (MP).")

        # Compute Buttoms
        self.bottomRightPanel.add_toolbar_buttons({
            "Compute Grids": self._btn("Compute Grids"),
            "Reset": self._btn("Reset")
        })
        self.bottomRightPanel.toolbarButtons["Compute Grids"].setToolTip(
            "Aktuelle Annotations sammeln und Gitter berechnen.")
        self.bottomRightPanel.toolbarButtons["Reset"].setToolTip("Alles im Compute-Panel zurücksetzen.")

    def _setup_drop_areas(self):
        # per panel an own ImageDropArea-Instance
        self.dropHighRes = ImageDropArea("Drag&Drop Image here")
        self.dropSD = ImageDropArea("Drag&Drop Image here")
        self.dropMicro = ImageDropArea("Drag&Drop Image here")

        # Left: Set Drop Areas directly as content-Area
        self.topLeftPanel.set_content(self.dropHighRes)
        self.bottomLeftPanel.set_content(self.dropSD)
        self.topRightPanel.set_content(self.dropMicro)

        # Rechts: Content-Container mit DropArea (oben) + Bottom-Bar (unten, 100px)
        self.rightContent = QWidget()
        v = QVBoxLayout(self.rightContent)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # (Optional) zum Testen:
        self.dropHighRes.imageDropped.connect(lambda p: print("[HighRes] Dropped:", p))
        self.dropSD.imageDropped.connect(lambda p: print("[SD] Dropped:", p))
        self.dropMicro.imageDropped.connect(lambda p: print("[Micro] Dropped:", p))

    def _connect_slider_signals(self):
        # Connect Slider signals
        # High Res Panel
        if self.topLeftPanel.contrastSlider:
            self.topLeftPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[HighRes] Contrast -> {v}")
            )
        if self.topLeftPanel.brightnessSlider:
            self.topLeftPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[HighRes] Brightness -> {v}")
            )

        # SD OCT Panel
        if self.bottomLeftPanel.contrastSlider:
            self.bottomLeftPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[SD] Contrast -> {v}")
            )
        if self.bottomLeftPanel.brightnessSlider:
            self.bottomLeftPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[SD] Brightness -> {v}")
            )

        # Microperimetry Panel
        if self.topRightPanel.contrastSlider:
            self.topRightPanel.contrastSlider.valueChanged.connect(
                lambda v: print(f"[Micro] Contrast -> {v}")
            )
        if self.topRightPanel.brightnessSlider:
            self.topRightPanel.brightnessSlider.valueChanged.connect(
                lambda v: print(f"[Micro] Brightness -> {v}")
            )

    @staticmethod
    def _btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        return btn

    def _connect_draw_seg(self):
        for panel in (self.topLeftPanel, self.bottomLeftPanel, self.topRightPanel):
            btn = panel.toolbarButtons.get("Draw Seg")
            if btn:
                btn.clicked.connect(self.clicked)

    def clicked(self):
        print("Button wurde geklickt!")