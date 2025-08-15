from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton
from PyQt6.QtCore import Qt
from .dropArea import ImageDropArea
from Controller.MPController import ImageController
from .panel import Panel


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

        # Starting panel window size + resizing panels
        leftSplitter.setSizes([200, 200])
        rightSplitter.setSizes([350, 50])
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

        # Compute Buttoms
        self.bottomRightPanel.add_toolbar_buttons({
            "Compute Grids": self._btn("Compute Grids"),
            "Reset": self._btn("Reset")
        })

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