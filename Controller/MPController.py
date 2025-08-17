from __future__ import annotations
from typing import Dict
from Model.image_ops import apply_contrast_brightness
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable, QTimer
from PyQt6.QtGui import QImage, QImageReader
from Controller.enums import MouseStatus
from Model.image_state import ImageState
from Model.seg_ops import deform_points_gaussian, laplacian_smooth, build_edit_window, nearest_seg_point_index

# Worker infrastructure: Needed because it allows parallel threading parallel to the GUI-Thread by
# performing calculations in the background.
# Advantages: Prevent threads from blocking the GUI-Thread, perform calculations in the background and
# send them back to the controller via queued connections (threadsafe)

# This Class (extends the QObject class) has a signal that contains a pyqt-signal.
# it when a worker is done, he releases a finish-emit, specifying its panel, its version and the image that is processed.
class _ResultSignal(QObject):
    finished = pyqtSignal(str, int, QImage)


# QRunnable is a job for the QThreadPool - it is created and given by calling pool.start(task). Qt then runs
# this task in the background-task, and it eases up the process of how the slider works.
# noinspection PyUnresolvedReferences
class _ProcessTask(QRunnable):
    def __init__(self, panel_id: str, version: int, qimg: QImage, c: int, b: int, sig: _ResultSignal):
        super().__init__()
        self.panel_id = panel_id # To which panel does this task belong?
        self.version = version # The Version
        self.qimg = qimg # the original image
        self.c = c # contrast value
        self.b = b # brightness value
        self.sig = sig # signal object to emit
        self._hr_edit = None

    def run(self):
        out = apply_contrast_brightness(self.qimg, self.c, self.b)
        self.sig.finished.emit(self.panel_id, self.version, out)


# --- Controller ---
class ImageController(QObject):
    def __init__(self, view):
        super().__init__()
        self.view = view # References the view the controller is responsible for.
        # Important for accessing widgets (Drop areas, slider, buttons) and wiring them.
        self.pool = QThreadPool.globalInstance() # Global threadPool for background Jobs

        # For each panel, we define an own Model-state: Own path, slider, version, segmentation / point lists,
        # so that we can apply the same logic to each of the panels.
        self.states: Dict[str, ImageState] = {
            "highres": ImageState(),
            "sd": ImageState(),
            "micro": ImageState(),
        }
        self.sig = _ResultSignal() # The QObject that contains the PyQtSignal
        self.sig.finished.connect(self._on_task_finished) # "Bridge" from the Worker back to the GUI-Thread

        # Debounce-Timer per panel - slider fire continuously, therefore wait 30ms before
        # processing slider position
        self._debounce: Dict[str, QTimer] = {}
        for pid in self.states.keys():
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(30)  # ms
            t.timeout.connect(lambda pid=pid: self._launch_processing(pid))
            self._debounce[pid] = t

        # Parameters for editing segmentation
        self.edit_R_screen = 30.0
        self.edit_strength = 1.0
        self.edit_sigma = self.edit_R_screen * 0.5
        self._edit_hit_radius = 8.0  # the radius of the circular area where we can "grab" the segmentation line and
        # drag it around

        # Stores the interaction mode per panel
        self.mode: Dict[str, MouseStatus] = {
            "highres": MouseStatus.IDLE,
            "sd": MouseStatus.IDLE,
            "micro": MouseStatus.IDLE,
        }

        # Stores the last image coordinate when editing, important for editing the segmentation line
        self._edit_last_xy: Dict[str, tuple[float, float] | None] = {
            "highres": None, "sd": None, "micro": None
        }

        # Wires all UI Signals with their respective Controller slots
        self._wire_view()

    # Wiring - connecting the view (widgets, buttons) with the logic
    def _wire_view(self):
        v = self.view # Convenient access to all Panels, Drop areas, buttons or sliders

        # Image Drops for all 3 panels
        #   ImageDropArea.dropEvent() emits imageDropped(path)-pyqt-Signal.
        #   Via lambda, the panel_id is given to the function load_image
        #   Loadimage reads the QImage, updates ImageState and updates the view
        v.dropHighRes.imageDropped.connect(lambda p: self.load_image("highres", p))
        v.dropSD.imageDropped.connect(lambda p: self.load_image("sd", p))
        v.dropMicro.imageDropped.connect(lambda p: self.load_image("micro", p))

        # Connect the sliders
        v.topLeftPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("highres", val))
        v.topLeftPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("highres", val))
        v.bottomLeftPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("sd", val))
        v.bottomLeftPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("sd", val))
        v.topRightPanel.contrastSlider.valueChanged.connect(lambda val: self.on_contrast("micro", val))
        v.topRightPanel.brightnessSlider.valueChanged.connect(lambda val: self.on_brightness("micro", val))

        # There is always the same variable btn used since
        # Connect the reset buttons
        # btn a temporary var that is used to get the button object and connect it to the responsible function
        btn = v.topLeftPanel.toolbarButtons.get("Reset")
        btn.clicked.connect(lambda: self.reset_adjustments("highres"))
        btn = v.bottomLeftPanel.toolbarButtons.get("Reset")
        btn.clicked.connect(lambda: self.reset_adjustments("sd"))
        btn = v.topRightPanel.toolbarButtons.get("Reset")
        btn.clicked.connect(lambda: self.reset_adjustments("micro"))

        # Other Buttons for High Res OCT
        btn = v.topLeftPanel.toolbarButtons.get("Draw Seg")
        btn.clicked.connect(lambda: self._draw_seg_activate("highres"))
        btn = v.topLeftPanel.toolbarButtons.get("Draw Pts")
        btn.clicked.connect(lambda: self._draw_pts_activate("highres"))
        btn = v.topLeftPanel.toolbarButtons.get("Edit Seg")
        btn.clicked.connect(lambda: self._edit_seg_activate("highres"))
        btn = v.topLeftPanel.toolbarButtons.get("Del Str")
        btn.clicked.connect(lambda: self._del_str_activate("highres"))

        v.dropHighRes.segDrawStart.connect(lambda x, y: self._seg_start("highres", x, y))
        v.dropHighRes.segDrawMove.connect(lambda x, y: self._seg_move("highres", x, y))
        v.dropHighRes.segDrawEnd.connect(lambda x, y: self._seg_end("highres", x, y))
        v.dropHighRes.pointAdded.connect(lambda x, y: self._point_added("highres", x, y))
        v.dropHighRes.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("highres", x1, y1, x2, y2))
        v.dropHighRes.segEditStart.connect(lambda x, y: self._edit_start("highres", x, y))
        v.dropHighRes.segEditMove.connect(lambda x, y: self._edit_move("highres", x, y))
        v.dropHighRes.segEditEnd.connect(lambda x, y: self._edit_end("highres", x, y))

        # ------ FOR SD OCT ------
        btn = v.bottomLeftPanel.toolbarButtons.get("Draw Seg")
        btn.clicked.connect(lambda: self._draw_seg_activate("sd"))
        btn = v.bottomLeftPanel.toolbarButtons.get("Draw Pts")
        btn.clicked.connect(lambda: self._draw_pts_activate("sd"))
        btn = v.bottomLeftPanel.toolbarButtons.get("Edit Seg")
        btn.clicked.connect(lambda: self._edit_seg_activate("sd"))
        btn = v.bottomLeftPanel.toolbarButtons.get("Del Str")
        btn.clicked.connect(lambda: self._del_str_activate("sd"))

        v.dropSD.segDrawStart.connect(lambda x, y: self._seg_start("sd", x, y))
        v.dropSD.segDrawMove.connect(lambda x, y: self._seg_move("sd", x, y))
        v.dropSD.segDrawEnd.connect(lambda x, y: self._seg_end("sd", x, y))
        v.dropSD.pointAdded.connect(lambda x, y: self._point_added("sd", x, y))
        v.dropSD.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("sd", x1, y1, x2, y2))
        v.dropSD.segEditStart.connect(lambda x, y: self._edit_start("sd", x, y))
        v.dropSD.segEditMove.connect(lambda x, y: self._edit_move("sd", x, y))
        v.dropSD.segEditEnd.connect(lambda x, y: self._edit_end("sd", x, y))

        # ------ FOR MICRO (MP) ------
        btn = v.topRightPanel.toolbarButtons.get("Draw Pts")
        btn.clicked.connect(lambda: self._draw_pts_activate("micro"))
        btn = v.topRightPanel.toolbarButtons.get("Del Pts")
        btn.clicked.connect(lambda: self._del_str_activate("micro"))

        v.dropMicro.pointAdded.connect(lambda x, y: self._point_added("micro", x, y))
        v.dropMicro.deleteRect.connect(lambda x1, y1, x2, y2: self._delete_rect("micro", x1, y1, x2, y2))

    # Draw Points Functionality
    def _draw_pts_activate(self, panel_id: str):
        self._set_status(panel_id, MouseStatus.DRAW_PTS)

    def _point_added(self, panel_id: str, x: float, y: float):
        st = self.states[panel_id]
        st.pts_points.append((x, y)) # Append drawn points to the point list
        self._drop_of(panel_id).set_points(st.pts_points) # Update the view - internally call update(), which calls paintEvent()

    # Draw Segmentation Functionality
    def _draw_seg_activate(self, panel_id: str):
        self.states[panel_id].seg_points.clear() # Allows for only one segmentation line to be drawn.
        self._set_status(panel_id, MouseStatus.DRAW_SEG)

    def _seg_start(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points = [(x, y)] # Add the starting point to the segmentation point list
        self._drop_of(panel_id).set_segmentation(st.seg_points) # Update the view

    def _seg_move(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points.append((x, y)) # Add the points the mouse pointer reaches during segmentation
        # drawing movement to the list of segmentation points
        self._drop_of(panel_id).set_segmentation(st.seg_points) # Update the view

    def _seg_end(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.DRAW_SEG): return
        st = self.states[panel_id]
        st.seg_points.append((x, y))
        self._drop_of(panel_id).set_segmentation(st.seg_points)

    # Delete Structures functionality
    def _del_str_activate(self, panel_id: str):
        self._set_status(panel_id, MouseStatus.DEL_STR)

    def _delete_rect(self, panel_id: str, x1: float, y1: float, x2: float, y2: float):
        # Activated when the mouse is released after drawing the rectangle
        # Set the coordinates of the starting and end point in image coordinates
        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
        # Get the affected panel
        st = self.states[panel_id]

        # Get the before-deletion-state of the points
        st.pts_points = [
            (x, y) for (x, y) in st.pts_points
            if not (xmin <= x <= xmax and ymin <= y <= ymax)
            # Set a new list: Only those points who are NOT in the delete rectangle stay in the list. update the list.
        ]

        # clear the segmentation list only if ALL segmentation points lie within the delete rectangle
        if st.seg_points and all(xmin <= x <= xmax and ymin <= y <= ymax for (x, y) in st.seg_points):
            st.seg_points.clear()

        # Update the view, give the updated segmentation + point list to the affected ImageDropAra
        # internally calls update(), which calls paintEvent()
        drop = self._drop_of(panel_id)
        if drop:
            drop.set_points(st.pts_points)
            drop.set_segmentation(st.seg_points)

    # Edit Segmentation Functionality
    def _edit_seg_activate(self, panel_id: str):
        # When the Edit-Seg Button is clicked: Set status of the Mouse to EDIT_SEG Mode
        self._set_status(panel_id, MouseStatus.EDIT_SEG)

    def _edit_start(self, panel_id: str, x: float, y: float):
        # Only allow start of segmentation when the EDIT_SEG Mode is activated.
        if not self._status_is(panel_id, MouseStatus.EDIT_SEG): return
        # Find the next base point of the actual segmentation to the click position
        idx, dist = nearest_seg_point_index(self.states[panel_id].seg_points, x, y)
        if idx is None: return # If there is no segmentation: There is nothing to edit.
        # Hit Test: The Mouseclick has a "grab" radius of 8px - if the segmentation line lies within these 8 pixels, the
        # line is "grabbed" and can be edited
        drop = self._drop_of(panel_id)
        # Secures that no matter what zoom the segmentation radius always stays same
        hit_r_img = self._edit_hit_radius / max(1e-6, drop.current_scale())
        if dist > hit_r_img: return
        # Store the starting point of the segmentation edit
        self._edit_last_xy[panel_id] = (x, y)

    def _edit_move(self, panel_id: str, x: float, y: float):
        if not self._status_is(panel_id, MouseStatus.EDIT_SEG):
            return
        # Without _edit_start (which sets the base point for segmentation edit) there will no valid drag be performed
        last = self._edit_last_xy.get(panel_id)
        if last is None:
            return

        lx, ly = last
        # The curve is moved incrementally (with each signal) by dx, dy
        dx, dy = x - lx, y - ly
        # If there is no movement: Return - saves work
        if dx == 0 and dy == 0:
            return

        # Get the actual segmentation
        st = self.states[panel_id]
        # When there is no segmentation, then there is nothing to deform.
        pts = st.seg_points
        if not pts:
            return

        # Radius of the brush size dependent on the zoom factor
        drop = self._drop_of(panel_id)
        s = drop.current_scale()
        R_img = self.edit_R_screen / s  # Brush Radius in image pixels
        sigma = max(1.0, R_img * 0.5)

        # Apply the deform algorithm (the edit segmentation functionality)
        deform_points_gaussian(
            pts=pts,
            center=(x, y),
            delta=(dx, dy),
            radius=R_img,
            strength=self.edit_strength,
            sigma=sigma,
        )
        # Flatten the segmentation edit
        laplacian_smooth(pts, iters=1, lam=0.2 * (1.0 / max(1.0, s)) ** 0.3)

        # Update the segmentation
        drop.set_segmentation(pts)
        self._edit_last_xy[panel_id] = (x, y)

    def _edit_end(self, panel_id: str, x: float, y: float):
        self._edit_last_xy[panel_id] = None

    # Reset functionality
    def reset_adjustments(self, panel_id: str):
        st = self.states[panel_id]
        self._debounce[panel_id].stop() # Stops Debounce timer
        # If a slider-movement is being registered, this process is terminated
        # prevent overwriting the reset with another slider-value changer
        st.contrast = 150
        st.brightness = 0 # Reset slider values
        st.version += 1 # Incrase version number of the panel

        # Give this information to the affected panel
        panel = self._panel_of(panel_id)
        if panel and panel.contrastSlider:
            panel.contrastSlider.blockSignals(True)
            panel.contrastSlider.setValue(150)
            panel.contrastSlider.blockSignals(False)
        if panel and panel.brightnessSlider:
            panel.brightnessSlider.blockSignals(True)
            panel.brightnessSlider.setValue(0)
            panel.brightnessSlider.blockSignals(False)

        # Show original image
        if st.original is not None:
            drop = self._drop_of(panel_id)
            if drop and hasattr(drop, "show_qimage"):
                drop.show_qimage(st.original)

        # Reset Mouse status to IDLE
        self._set_status(panel_id, MouseStatus.IDLE, cursor=False)

    # Helper functions for image-itself display and editing (via the contrast and brightness sliders)
    def load_image(self, panel_id: str, path: str):
        # Reads an image with the QImageReader
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()

        st = self.states[panel_id]
        st.path = path
        st.original = img  # Writes the image into the Model-state of the panel
        self._schedule(panel_id)  # Gives the job to the scheduler to start Image display

    def on_contrast(self, panel_id: str, value: int):
        st = self.states[panel_id]
        st.contrast = value # Update the state with the new contrast value
        self._schedule(panel_id) # Giv the job to the scheduler to update image display with contrast

    def on_brightness(self, panel_id: str, value: int):
        st = self.states[panel_id]
        st.brightness = value
        self._schedule(panel_id)

    # Status and Panel Activation
    def _set_status(self, panel_id: str, status: MouseStatus, *, cursor=True):
        self.mode[panel_id] = status # Set the statuas
        drop = self._drop_of(panel_id)
        if drop: # Update Mouse Status and Cursor
            drop.set_mouse_status(status)
            drop.set_draw_cursor(cursor)

    def _status_is(self, panel_id: str, status: MouseStatus) -> bool:
        return self.mode.get(panel_id, MouseStatus.IDLE) == status

    # Helper methods
    def _schedule(self, panel_id: str):
        # Debounce: Wait before starting the job to prevent "job"-spamming, start Debounce-Timer
        self._debounce[panel_id].start()

    def _launch_processing(self, panel_id: str):
        # Called by the Debounce-timer
        # Check whether the original image exists
        st = self.states[panel_id]
        if st.original is None:
            return
        # Increment Version (Everything before is thus marked as "old")
        st.version += 1
        # Start a new Task with affected panel_ds and the relevant information
        task = _ProcessTask(panel_id, st.version, st.original, st.contrast, st.brightness, self.sig)
        self.pool.start(task) # Start the worker in the threadpool

    def _on_task_finished(self, panel_id: str, version: int, qimg: QImage):
        st = self.states[panel_id]
        # Ignore old updates / jobs
        if version != st.version:
            return
        # return result
        drop = self._drop_of(panel_id)
        if drop:
            if hasattr(drop, "show_qimage"):
                drop.show_qimage(qimg)
            else:
                from PyQt6.QtGui import QPixmap
                drop.setPixmap(QPixmap.fromImage(qimg))
                drop.setText("")

    def _panel_of(self, panel_id: str):
        # Maps desired panel
        v = self.view
        return {"highres": v.topLeftPanel, "sd": v.bottomLeftPanel, "micro": v.topRightPanel}.get(panel_id)

    def _drop_of(self, panel_id: str):
        # Maps desired ImageDropArea
        v = self.view
        return {"highres": v.dropHighRes, "sd": v.dropSD, "micro": v.dropMicro}.get(panel_id)