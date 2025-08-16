from PyQt6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy, QPushButton, QToolButton, QSlider
)
from PyQt6.QtCore import Qt

class Panel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        # Title
        self.titleLabel = QLabel(title)
        self.titleLabel.setContentsMargins(4, 4, 4, 4)
        self.titleLabel.setFixedHeight(20)
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        # Toolbar
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(50)
        self._tbLayout = QHBoxLayout(self.toolbar)
        self._tbLayout.setContentsMargins(8, 4, 8, 4)
        self._tbLayout.setSpacing(8)
        self._tbLayout.addStretch(1)

        # Content
        self.contentArea = QWidget()
        self.contentArea.setObjectName("contentArea")
        self.contentArea.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Build the panel layout
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.titleLabel)
        v.addWidget(self.toolbar)
        v.addWidget(self.contentArea)

        # All buttons are stored in a dictionary and can be accessed later as objects
        self.toolbarButtons: dict[str, QPushButton | QToolButton] = {}
        self.contrastSlider: QSlider | None = None
        self.brightnessSlider: QSlider | None = None

    # ---------- Public API ---------
    def add_toolbar_buttons(self, buttons: dict[str, QPushButton | QToolButton]):
        # Buttons are added to the dict

        # The last item of the layout is the stretch - take away
        stretch_item = self._tbLayout.takeAt(self._tbLayout.count() - 1)

        # add buttons
        for key, btn in buttons.items():
            if btn.minimumHeight() < 36:
                btn.setMinimumHeight(36)
            self._tbLayout.addWidget(btn)
            self.toolbarButtons[key] = btn

        # re-add the stretch
        self._tbLayout.addItem(stretch_item)

    def add_adjustment_sliders(
        self,
        with_labels: bool = True,
        contrast_range: tuple[int, int] = (0, 300),
        contrast_default: int = 150,
        brightness_range: tuple[int, int] = (-150, 150),
        brightness_default: int = 0,
        slider_min_width: int = 120,
    ):
        if self.contrastSlider is not None or self.brightnessSlider is not None:
            return  # alrady added

        stretch_item = self._tbLayout.takeAt(self._tbLayout.count() - 1)

        # add the labels of the sliders
        if with_labels:
            lbl_c = QLabel("Contrast")
            lbl_c.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._tbLayout.addWidget(lbl_c)

        # Contrast-slider
        self.contrastSlider = QSlider(Qt.Orientation.Horizontal)
        self.contrastSlider.setRange(*contrast_range)
        self.contrastSlider.setValue(contrast_default)
        self.contrastSlider.setSingleStep(1)
        self.contrastSlider.setMinimumHeight(36)
        self.contrastSlider.setMinimumWidth(slider_min_width)
        self.contrastSlider.setObjectName("ContrastSlider")
        self._tbLayout.addWidget(self.contrastSlider)

        if with_labels:
            lbl_b = QLabel("Brightness")
            lbl_b.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._tbLayout.addWidget(lbl_b)

        # Brightness-slider
        self.brightnessSlider = QSlider(Qt.Orientation.Horizontal)
        self.brightnessSlider.setRange(*brightness_range)
        self.brightnessSlider.setValue(brightness_default)
        self.brightnessSlider.setSingleStep(1)
        self.brightnessSlider.setMinimumHeight(36)
        self.brightnessSlider.setMinimumWidth(slider_min_width)
        self.brightnessSlider.setObjectName("BrightnessSlider")
        self._tbLayout.addWidget(self.brightnessSlider)

        # Add the stretch again to the end to fill up the remaining space
        self._tbLayout.addItem(stretch_item)

    def set_content(self, widget: QWidget):
        # Replaces the content through own widget
        layout = self.layout()
        layout.removeWidget(self.contentArea)
        self.contentArea.deleteLater()
        self.contentArea = widget
        self.contentArea.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.contentArea)

    def get_button(self, key: str):
        return self.toolbarButtons.get(key)