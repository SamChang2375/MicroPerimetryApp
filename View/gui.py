from .dropArea import ImageDropArea
from Controller.MPController import ImageController
from .panel import Panel
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QPushButton, QLineEdit, QSizePolicy
from PyQt6.QtCore import Qt, QTimer

class MicroPerimetryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroPerimetryApp")

        # Initialize the layout of the app
        self._init_ui()

        # Initialize the controller. The controller controls all interactions between the view (gui.py)
        # and the model (package Model). self as parameter (= MicroperimetryGUI instance) so that the controller
        # can access all buttons, panels or drop areas of the app, connect signals etc.
        self.controller = ImageController(self)

    # This method initializes the layout of the app.
    def _init_ui(self):
        # Window Background
        palette = self.palette()
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setMinimumSize(900, 600)

        # Build Splitter structure
        mainSplitter = QSplitter(Qt.Orientation.Horizontal)   # left / right
        leftSplitter = QSplitter(Qt.Orientation.Vertical)      # upper / lower
        rightSplitter = QSplitter(Qt.Orientation.Vertical)
        self.rightSplitter = rightSplitter

        # create four Panels --> Implemented in the Panel-Class
        self.topLeftPanel = Panel("High Res OCT Image")
        self.bottomLeftPanel = Panel("SD OCT Image")
        self.topRightPanel = Panel("MicroPerimetry Imabge")
        self.bottomRightPanel = Panel("Compute Area")

        # Put the panels together to the Layout
        leftSplitter.addWidget(self.topLeftPanel)
        leftSplitter.addWidget(self.bottomLeftPanel)
        rightSplitter.addWidget(self.topRightPanel)

        # bottom right panel: limit its height to 160px
        rightSplitter.addWidget(self.bottomRightPanel)
        MAX_H = 160
        self.bottomRightPanel.setMaximumHeight(MAX_H)
        # bottom right panel may grow horizontally, but not vertically larger than MAX_H
        self.bottomRightPanel.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Fixed)
        # Set the startup panel sizes
        QTimer.singleShot(0, lambda: self.rightSplitter.setSizes([
            max(self.rightSplitter.size().height() - MAX_H, 0),  # oben
            MAX_H  # unten
        ]))

        # Putting it altogether
        mainSplitter.addWidget(leftSplitter)
        mainSplitter.addWidget(rightSplitter)
        # Set the stretch factors:
        mainSplitter.setStretchFactor(0, 1)  # left
        mainSplitter.setStretchFactor(1, 2)  # right
        leftSplitter.setStretchFactor(0, 1) # High Res OCT
        leftSplitter.setStretchFactor(1, 1) # SD OCT
        rightSplitter.setStretchFactor(0, 3)  # MP
        rightSplitter.setStretchFactor(1, 1)  # Compute Grid Panel

        # Set Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(mainSplitter)

        # Set up the toolbars and the Image-Drop-Areas
        self._setup_toolbars()
        self._setup_drop_areas()
        self._setup_compute_controls()

    # Functions to build up interaction functionalities
    def _setup_toolbars(self):
        # HighRes Buttons
        self.topLeftPanel.add_toolbar_buttons({
            "Draw Seg": self._btn("Draw Seg"),
            "Draw Pts": self._btn("Draw Pts"),
            "Edit Seg": self._btn("Edit Seg"),
            "Del Str": self._btn("Del Str"),
            "Reset": self._btn("Reset")
        })
        # High Res Button tooltips
        self.topLeftPanel.toolbarButtons["Draw Seg"].setToolTip(
            "Draw the Segmentation line by left-click holding the mouse. \n"
            "Release the mouse when finished with drawing.")
        self.topLeftPanel.toolbarButtons["Draw Pts"].setToolTip(
            "Draw points by left-clicking on the desired spot where to \n"
            "draw the point. It is necessary to spread out the points over the whole image \n"
            "so that the matching algorithm for grid computung works best. \n"
            "Mark at least 4 points, optimal are 7 points. Use the arterial structures \n"
            "to find corresponding points in each of the three image. \n"
            "Each image must have the same number of drawn points!")
        self.topLeftPanel.toolbarButtons["Edit Seg"].setToolTip(
            "Edit the segmentation line by left-click-holding the mouse, \n"
            "thus 'grabbing' the desired Section of the Segmentation line to edit. \n"
            "this funcionality allows one to smoothen out the segmentation line and \n"
            "correct missegmented parts.")
        self.topLeftPanel.toolbarButtons["Del Str"].setToolTip(
            "Draw a Rectangle by left-click-holding on the image, dragging and releasing \n"
            "the mouse-click. All structures (points and segmentation lines) that lie within \n"
            "the drawn rectangle will be deleted.")
        self.topLeftPanel.toolbarButtons["Reset"].setToolTip(
            "Resets the image to original and resets the mouse mode. That means: \n"
            "If one wants to start drawing points again after clicking the reset button, \n"
            "the 'Draw Pts'-button needs to be clicked again.")
        # High Res Sliders
        self.topLeftPanel.add_adjustment_sliders(
            with_labels=True,  # Show labels "Contrast" and "Brightness"
            contrast_range=(0, 300),  # 150 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140
        )

        # SD OCT Buttons
        self.bottomLeftPanel.add_toolbar_buttons({
            "Draw Seg": self._btn("Draw Seg"),
            "Draw Pts": self._btn("Draw Pts"),
            "Edit Seg": self._btn("Edit Seg"),
            "Del Str": self._btn("Del Str"),
            "Reset": self._btn("Reset")
        })
        # SD OCT Button tooltips
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
        # SD OCT Sliders
        self.bottomLeftPanel.add_adjustment_sliders(
            with_labels=True,  # Show labels "Contrast" and "Brightness"
            contrast_range=(0, 300),  # 150 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140
        )

        # Microperimetry Buttons
        self.topRightPanel.add_toolbar_buttons({
            "Draw Pts": self._btn("Draw Pts"),
            "Del Pts": self._btn("Del Pts"),
            "Reset": self._btn("Reset")
        })
        # Microperimetry Button tooltips
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
        #
        self.topRightPanel.add_adjustment_sliders(
            with_labels=True,  # Show labels "Contrast" and "Brightness"
            contrast_range=(0, 300),  # 150 = neutral
            contrast_default=150,
            brightness_range=(-150, 150),
            brightness_default=0,
            slider_min_width=140
        )

        # Compute Grid Buttons
        self.bottomRightPanel.add_toolbar_buttons({
            "Comp Grids AppSeg": self._btn("Compute Grids AppSeg"),
            "Comp Grids PreSeg": self._btn("Compute Grids PreSeg"),
            "Reset": self._btn("Reset")
        })
        # Compute Grid Button tooltips
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
        # Create 3 Drop aras - one per Image window panel.
        self.dropHighRes = ImageDropArea("Drag&Drop Image here")
        self.dropSD = ImageDropArea("Drag&Drop Image here")
        self.dropMicro = ImageDropArea("Drag&Drop Image here")

        # Left: Set Drop Areas directly as content-Area
        self.topLeftPanel.set_content(self.dropHighRes)
        self.bottomLeftPanel.set_content(self.dropSD)
        self.topRightPanel.set_content(self.dropMicro)

    def _setup_compute_controls(self):
        container = QWidget()
        v = QVBoxLayout(container)

        # Line-Edit to edit patient ID. Important for the naming of the result files after compute-grid-analysis.
        self.computeInput = QLineEdit()
        self.computeInput.setPlaceholderText("Enter Patient ID number! e.g. '22'")
        self.computeInput.setClearButtonEnabled(True)
        self.computeInput.setFixedHeight(24)
        self.computeInput.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        v.addWidget(self.computeInput)

        # "Console" which after computing grids shows the folder in which the results (e.g. the XML File with the coordinates) is located.
        self.computeStatus = QLineEdit()
        self.computeStatus.setReadOnly(True)
        self.computeStatus.setPlaceholderText("This field will show the location of the folder in which the computation results are stored")
        self.computeStatus.setFixedHeight(22)
        self.computeStatus.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.computeStatus.setStyleSheet("QLineEdit { background:#1e1e1e; color:#d8d8d8; padding:2px 6px; }")
        v.addWidget(self.computeStatus)

        self.bottomRightPanel.set_content(container)

    # "Recipe" to create buttons of the sam style
    @staticmethod
    def _btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        return btn