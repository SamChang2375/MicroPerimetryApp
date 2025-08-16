from PyQt6.QtCore import Qt
from .dropArea import ImageDropArea
from Controller.MPController import ImageController
from .panel import Panel
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLineEdit, QSizePolicy
from PyQt6.QtCore import Qt, QTimer


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
        self.rightSplitter = rightSplitter

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

        # --- Bottom-Right-Panel deckeln ---
        MAX_H = 160  # gewünschte Maximalhöhe (px), ggf. anpassen

        # entweder maximale Höhe ...
        self.bottomRightPanel.setMaximumHeight(MAX_H)
        # ... oder fixe Höhe (dann ist es immer exakt so hoch):
        # self.bottomRightPanel.setFixedHeight(MAX_H)

        # Panel soll horizontal wachsen dürfen, vertikal nicht
        self.bottomRightPanel.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Fixed)

        # oben soll den restlichen Platz bekommen
        self.rightSplitter.setStretchFactor(0, 1)
        self.rightSplitter.setStretchFactor(1, 0)

        # Anfangs-Sizes setzen, sobald Geometrie steht
        QTimer.singleShot(0, lambda: self.rightSplitter.setSizes([
            max(self.rightSplitter.size().height() - MAX_H, 0),  # oben
            MAX_H  # unten
        ]))

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
        self._setup_compute_controls()
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
        self.topLeftPanel.toolbarButtons["Draw Seg"].setToolTip("Draw the Segmentation line by left-click holding the mouse. \n"
                                                                "Release the mouse when finished with drawing.")
        self.topLeftPanel.toolbarButtons["Draw Pts"].setToolTip("Draw points by left-clicking on the desired spot where to \n"
                                                                "draw the point. It is necessary to spread out the points over the whole image \n"
                                                                "so that the matching algorithm for grid computung works best. \n"
                                                                "Mark at least 4 points, optimal are 7 points. Use the arterial structures \n"
                                                                "to find corresponding points in each of the three image. \n"
                                                                "Each image must have the same number of drawn points!")
        self.topLeftPanel.toolbarButtons["Edit Seg"].setToolTip("Edit the segmentation line by left-click-holding the mouse, \n"
                                                                "thus 'grabbing' the desired Section of the Segmentation line to edit. \n"
                                                                "this funcionality allows one to smoothen out the segmentation line and \n"
                                                                "correct missegmented parts.")
        self.topLeftPanel.toolbarButtons["Del Str"].setToolTip("Draw a Rectangle by left-click-holding on the image, dragging and releasing \n"
                                                               "the mouse-click. All structures (points and segmentation lines) that lie within \n"
                                                               "the drawn rectangle will be deleted.")
        self.topLeftPanel.toolbarButtons["Reset"].setToolTip("Resets the image to original and resets the mouse mode. That means: \n"
                                                             "If one wants to start drawing points again after clicking the reset button, \n"
                                                             "the 'Draw Pts'-button needs to be clicked again.")

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
        self.bottomLeftPanel.toolbarButtons["Draw Seg"].setToolTip(
            "Draw the Segmentation line by left-click holding the mouse. \n"
            "Release the mouse when finished with drawing.")
        self.bottomLeftPanel.toolbarButtons["Draw Pts"].setToolTip(
            "Draw points by left-clicking on the desired spot where to \n"
            "draw the point. It is necessary to spread out the points over the whole image \n"
            "so that the matching algorithm for grid computung works best. \n"
            "Mark at least 4 points, optimal are 7 points. Use the arterial structures \n"
            "to find corresponding points in each of the three image. \n"
            "Each image must have the same number of drawn points!")
        self.bottomLeftPanel.toolbarButtons["Edit Seg"].setToolTip(
            "Edit the segmentation line by left-click-holding the mouse, \n"
            "thus 'grabbing' the desired Section of the Segmentation line to edit. \n"
            "this funcionality allows one to smoothen out the segmentation line and \n"
            "correct missegmented parts.")
        self.bottomLeftPanel.toolbarButtons["Del Str"].setToolTip(
            "Draw a Rectangle by left-click-holding on the image, dragging and releasing \n"
            "the mouse-click. All structures (points and segmentation lines) that lie within \n"
            "the drawn rectangle will be deleted.")
        self.bottomLeftPanel.toolbarButtons["Reset"].setToolTip(
            "Resets the image to original and resets the mouse mode. That means: \n"
            "If one wants to start drawing points again after clicking the reset button, \n"
            "the 'Draw Pts'-button needs to be clicked again.")

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
        self.topRightPanel.toolbarButtons["Draw Pts"].setToolTip(
            "Draw points by left-clicking on the desired spot where to \n"
            "draw the point. It is necessary to spread out the points over the whole image \n"
            "so that the matching algorithm for grid computung works best. \n"
            "Mark at least 4 points, optimal are 7 points. Use the arterial structures \n"
            "to find corresponding points in each of the three image. \n"
            "Each image must have the same number of drawn points!")
        self.topRightPanel.toolbarButtons["Del Pts"].setToolTip(
            "Draw a Rectangle by left-click-holding on the image, dragging and releasing \n"
            "the mouse-click. All points that lie within \n"
            "the drawn rectangle will be deleted."
        )
        self.topRightPanel.toolbarButtons["Reset"].setToolTip(
            "Resets the image to original and resets the mouse mode. That means: \n"
            "If one wants to start drawing points again after clicking the reset button, \n"
            "the 'Draw Pts'-button needs to be clicked again.")

        # Compute Buttoms
        self.bottomRightPanel.add_toolbar_buttons({
            "Comp Grids AppSeg": self._btn("Compute Grids AppSeg"),
            "Comp Grids PreSeg": self._btn("Compute Grids PreSeg"),
            "Reset": self._btn("Reset")
        })
        self.bottomRightPanel.toolbarButtons["Comp Grids AppSeg"].setToolTip(
            "Press this button after all segmentation lines and points are drawn. \n"
            "This button will then automatically compute the grids, show the resulting MP Grid, \n "
            "and saves the coordinates to an XML file \n"
            "that can be uploaded to the MAIA Microperimetry Device.")
        self.bottomRightPanel.toolbarButtons["Comp Grids PreSeg"].setToolTip(
            "Press this button after all points are drawn, and use it if one has already \n"
            "pre-segmented the DRIL areas in the OCT-Application \n"
            "This button will then automatically compute the grids, show the resulting MP Grid, \n "
            "and saves the coordinates to an XML file \n"
            "that can be uploaded to the MAIA Microperimetry Device.")
        self.bottomRightPanel.toolbarButtons["Reset"].setToolTip(
            "Resets the whole application: \n"
            "all images will be removed, all \n"
            "segmentation lines and points be deleted. \n"
            "Then, the next round of files can be uploaded \n"
            "for computing the grids for the next patient case.")

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

    from PyQt6.QtWidgets import QLineEdit, QSizePolicy

    def _setup_compute_controls(self):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 2, 4, 4)  # sehr kleine Außenränder
        v.setSpacing(2)  # ↓ Abstand zwischen Input & Status

        self.computeInput = QLineEdit()
        self.computeInput.setPlaceholderText("Enter Patient ID number! e.g. '22'")
        self.computeInput.setClearButtonEnabled(True)
        self.computeInput.setFixedHeight(24)
        self.computeInput.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        v.addWidget(self.computeInput)

        self.computeStatus = QLineEdit()
        self.computeStatus.setReadOnly(True)
        self.computeStatus.setPlaceholderText("This field will show the location of the folder in which the computation results are stored")
        self.computeStatus.setFixedHeight(22)
        self.computeStatus.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # (Optional) dezentes Styling
        self.computeStatus.setStyleSheet("QLineEdit { background:#1e1e1e; color:#d8d8d8; padding:2px 6px; }")
        v.addWidget(self.computeStatus)

        # als Content in das Panel
        self.bottomRightPanel.set_content(container)

    def _set_status(self, text: str):
        if hasattr(self, "computeStatus") and self.computeStatus:
            self.computeStatus.setText(text)