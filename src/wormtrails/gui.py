import os
import sys
import threading

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QLineEdit, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QFrame, QFileDialog,
    QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QFont

import wormtrails as wts
from wormtrails.processing import create_time_encoded_frame, create_time_encoded_frame_vectorized, fit_pixel_linear_model, subtract_average

__all__ = ['main']


class _ScheduleHelper(QObject):
    _signal = Signal(object)

    def __init__(self):
        super().__init__()
        self._signal.connect(self._run)

    def schedule(self, fn):
        self._signal.emit(fn)

    @staticmethod
    def _run(fn):
        fn()


_scheduler = _ScheduleHelper()


class ProgressLabel(QLabel):
    def set_text(self, text):
        _scheduler.schedule(lambda: self.setText(text))


class CollapsibleFrame(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._visible = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QWidget()
        hdr_layout = QVBoxLayout(hdr)
        hdr_layout.setContentsMargins(0, 0, 0, 0)
        hdr_layout.setSpacing(0)

        self._toggle_btn = QPushButton(f"[-] {title}")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; border: none; padding: 2px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
        )
        self._toggle_btn.clicked.connect(self.toggle)
        hdr_layout.addWidget(self._toggle_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        hdr_layout.addWidget(sep)

        layout.addWidget(hdr)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self._content)

    def toggle(self):
        self._visible = not self._visible
        self._content.setVisible(self._visible)
        text = self._toggle_btn.text()
        prefix = "[-]" if self._visible else "[+]"
        rest = text[4:] if len(text) > 4 else ""
        self._toggle_btn.setText(f"{prefix} {rest}")

    def content_layout(self):
        return self._content_layout

    def add_label_entry(self, label_text, row, col=1, width=10):
        grid = self._ensure_grid()
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(lbl, row, 0)
        entry = QLineEdit()
        entry.setMaximumWidth(80)
        entry.setText("")
        grid.addWidget(entry, row, col)
        return entry

    def add_checkbutton(self, label_text, row, default=False):
        grid = self._ensure_grid()
        cb = QCheckBox(label_text)
        cb.setChecked(default)
        grid.addWidget(cb, row, 1)
        return cb

    def add_combobox(self, label_text, values, row, default=None):
        grid = self._ensure_grid()
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(lbl, row, 0)
        cb = QComboBox()
        cb.addItems(values)
        if default and default in values:
            cb.setCurrentText(default)
        elif values:
            cb.setCurrentIndex(0)
        grid.addWidget(cb, row, 1)
        return cb

    def _ensure_grid(self):
        if not hasattr(self, '_grid'):
            self._grid = QGridLayout()
            self._grid.setHorizontalSpacing(4)
            self._grid.setVerticalSpacing(3)
            self._content_layout.addLayout(self._grid)
        return self._grid


def _safe_int(val, default):
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default

def _safe_float(val, default):
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


class WormtrailsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wormtrails Analysis & Visualization")
        self.resize(740, 720)

        self.video_path = ""
        self._last_count_assist_df = None

        self._video = None
        self._cached_video_path = ""
        self._corrected = None
        self._corrected_vig_key = None
        self._motion = None
        self._motion_key = None
        self._te_params = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 6, 10, 6)

        file_row = QHBoxLayout()
        file_row.setSpacing(4)
        self._path_label = QLabel("Video File:")
        file_row.addWidget(self._path_label)
        self._path_entry = QLineEdit()
        self._path_entry.setReadOnly(True)
        self._path_entry.setMinimumWidth(300)
        file_row.addWidget(self._path_entry)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(self._browse_btn)
        layout.addLayout(file_row)

        self.status_bar = QLabel("Ready")
        self.status_bar.setFrameStyle(QFrame.Sunken | QFrame.Panel)
        self.status_bar.setFixedHeight(24)
        layout.addWidget(self.status_bar)

        self.notebook = QTabWidget()
        layout.addWidget(self.notebook, 1)

        self.tab_vis = QWidget()
        self.notebook.addTab(self.tab_vis, "Visualizations")
        self.setup_visualization_tab()

        self.tab_assisted = QWidget()
        self.notebook.addTab(self.tab_assisted, "Computer Assisted")
        self.setup_assisted_tab()

        self.tab_auto = QWidget()
        self.notebook.addTab(self.tab_auto, "Computer Automated")
        self.setup_automated_tab()

        self.tab_chemotaxis = QWidget()
        self.notebook.addTab(self.tab_chemotaxis, "Measure Chemotaxis")
        self.setup_chemotaxis_tab()

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.avi *.mp4 *.mkv);;All Files (*.*)"
        )
        if path:
            self.video_path = path
            self._path_entry.setText(path)

    def _get_video(self):
        if not self.video_path:
            raise ValueError("No video file selected.")
        if self._video is None or self.video_path != self._cached_video_path:
            self._invalidate_caches()
            self._video = wts.read_video_file(self.video_path)
            self._cached_video_path = self.video_path
        return self._video

    def _ensure_corrected(self, vig_params):
        video = self._get_video()
        vig_key = str(sorted(vig_params.items()))
        if self._corrected is None or self._corrected_vig_key != vig_key:
            self._corrected = wts.correct_vignetting(video, **vig_params)
            self._corrected_vig_key = vig_key
            self._motion = None
            self._motion_key = None
        return self._corrected

    def _ensure_motion(self, corrected, method, sub_params):
        motion_key = (method, str(sorted(sub_params.items())))
        if self._motion is None or self._motion_key != motion_key:
            self._motion = self._compute_motion(corrected, method, sub_params)
            self._motion_key = motion_key
        return self._motion

    def _invalidate_caches(self):
        self._video = None
        self._corrected = None
        self._corrected_vig_key = None
        self._motion = None
        self._motion_key = None
        self._te_params = None

    def measure_action_wrapper(self, action_func, status_msg="Processing..."):
        if not self.video_path:
            QMessageBox.critical(self, "Error", "Please select a video file first.")
            return

        def run_thread():
            try:
                _scheduler.schedule(lambda: self.status_bar.setText(status_msg))
                action_func()
                _scheduler.schedule(lambda: self.status_bar.setText("Done"))
            except Exception as ex:
                _scheduler.schedule(lambda: self.status_bar.setText("Error"))
                _scheduler.schedule(lambda e=ex: QMessageBox.critical(
                    self, "Execution Error", str(e)))

        threading.Thread(target=run_thread, daemon=True).start()

    # --- Vis Tab Setup ---
    def setup_visualization_tab(self):
        layout = QVBoxLayout(self.tab_vis)

        desc = QLabel("Create previews of the video using different processing pipelines.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc)

        vig = CollapsibleFrame("Vignetting Correction")
        self._vig_kernel = vig.add_label_entry("Kernel Size (blank=auto):", 0)
        self._vig_median = vig.add_checkbutton("Use Median Blur", 1, default=False)
        layout.addWidget(vig)

        sub = CollapsibleFrame("Subtract Average / Motion Detection")
        self._vis_motion_method = sub.add_combobox(
            "Motion Method:", ["subtract_average", "linear_model_residuals"], 0,
            default="subtract_average")
        self._sub_start = sub.add_label_entry("Average Start:", 1)
        self._sub_start.setText("0")
        self._sub_end = sub.add_label_entry("Average End (blank=last):", 2)
        self._sub_abs = sub.add_checkbutton("Absolute Difference", 3, default=True)
        self._sub_proj = sub.add_checkbutton("Use Projection", 4)
        self._sub_light = sub.add_checkbutton("Light Background", 5, default=True)
        layout.addWidget(sub)

        te = CollapsibleFrame("Time Encoding")
        self._te_colormap = te.add_combobox(
            "Colormap:", [
                "black", "white", "blue_to_red", "white_to_black", "black_to_white",
                "banded_blue_to_red", "dark_separated_blue_to_red",
                "middle_grey_last_black", "hsv_rainbow"
            ], 0, default="black")
        self._te_window = te.add_label_entry("Window:", 1)
        self._te_window.setText("20")
        self._te_scale = te.add_label_entry("Scale Factor:", 2)
        self._te_scale.setText("1")
        self._te_offset = te.add_label_entry("Offset:", 3)
        self._te_offset.setText("0")
        self._te_start_frame = te.add_label_entry("Start Frame:", 4)
        self._te_light = te.add_checkbutton("Light Background", 5, default=True)
        self._te_vectorized = te.add_checkbutton("Use Vectorized Frame", 6)
        self._te_parallel = te.add_checkbutton("Use Parallel (save)", 7)
        layout.addWidget(te)

        self.vis_progress = ProgressLabel()
        layout.addWidget(self.vis_progress)

        btn_row = QHBoxLayout()
        btn_play = QPushButton("Play Original Video")
        btn_play.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_show_video, "Playing video..."))
        btn_row.addWidget(btn_play)

        btn_preview_frame = QPushButton("Preview Single Frame")
        btn_preview_frame.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_preview_single_frame, "Generating frame..."))
        btn_row.addWidget(btn_preview_frame)

        btn_te = QPushButton("Preview Time Encoding")
        btn_te.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_preview_time_encoding, "Creating preview..."))
        btn_row.addWidget(btn_te)

        btn_save_frame = QPushButton("Save Frame as Image")
        btn_save_frame.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_save_frame, "Saving frame..."))
        btn_row.addWidget(btn_save_frame)

        btn_save_video = QPushButton("Save as Video")
        btn_save_video.clicked.connect(self._do_save_video)
        btn_row.addWidget(btn_save_video)

        layout.addLayout(btn_row)
        layout.addStretch()

    def _get_vig_params(self):
        kernel = _safe_int(self._vig_kernel.text(), None)
        return {
            'kernel_size': kernel,
            'use_median_blur': self._vig_median.isChecked(),
        }

    def _get_vis_sub_params(self):
        start = _safe_int(self._sub_start.text(), 0)
        end = _safe_int(self._sub_end.text(), -1)
        return {
            'average_start': start,
            'average_end': end,
            'use_absolute_difference': self._sub_abs.isChecked(),
            'use_projection': self._sub_proj.isChecked(),
            'light_background': self._sub_light.isChecked(),
        }

    def _compute_motion(self, video, method, sub_params):
        if method == "linear_model_residuals":
            residuals, _, _ = fit_pixel_linear_model(video)
            residuals[residuals > 0] = 0
            residuals = residuals ** 2
            residuals[residuals > 255] = 255
            return residuals.astype(np.uint8)
        else:
            return subtract_average(video, **sub_params)

    def _get_te_params(self):
        colormap_map = {
            'black': wts.black,
            'white': wts.white,
            'blue_to_red': wts.blue_to_red,
            'white_to_black': wts.white_to_black,
            'black_to_white': wts.black_to_white,
            'banded_blue_to_red': wts.banded_blue_to_red,
            'dark_separated_blue_to_red': wts.dark_separated_blue_to_red,
            'middle_grey_last_black': wts.middle_grey_last_black,
            'hsv_rainbow': wts.hsv_rainbow,
        }
        cm_name = self._te_colormap.currentText()
        return {
            'colormap': colormap_map.get(cm_name, wts.blue_to_red),
            'window': _safe_int(self._te_window.text(), 20),
            'scale_factor': _safe_float(self._te_scale.text(), 1),
            'offset': _safe_int(self._te_offset.text(), 0),
            'light_background': self._te_light.isChecked(),
        }

    def _do_show_video(self):
        video = self._get_video()
        _scheduler.schedule(lambda v=video: wts.show_video_array(v))

    def _do_preview_time_encoding(self):
        vig = self._get_vig_params()
        corrected = self._ensure_corrected(vig)
        sub = self._get_vis_sub_params()
        method = self._vis_motion_method.currentText()
        self._motion = self._ensure_motion(corrected, method, sub)
        self._te_params = self._get_te_params()
        motion = self._motion
        te_params = self._te_params
        _scheduler.schedule(lambda m=motion, p=te_params: wts.show_time_encoding(m, **p))

    def _do_preview_single_frame(self):
        vig = self._get_vig_params()
        corrected = self._ensure_corrected(vig)
        sub = self._get_vis_sub_params()
        method = self._vis_motion_method.currentText()
        motion = self._ensure_motion(corrected, method, sub)
        te_params = self._get_te_params()
        start_frame = _safe_int(self._te_start_frame.text(), 0)
        frame = create_time_encoded_frame(
            motion,
            colormap=te_params['colormap'],
            window=te_params['window'],
            start_time=start_frame,
            scale_factor=te_params['scale_factor'],
            offset=te_params['offset'],
            light_background=te_params['light_background'],
        )
        _scheduler.schedule(lambda f=frame: wts.show_frame(f))

    def _do_save_frame(self):
        vig = self._get_vig_params()
        corrected = self._ensure_corrected(vig)
        sub = self._get_vis_sub_params()
        method = self._vis_motion_method.currentText()
        motion = self._ensure_motion(corrected, method, sub)
        te_params = self._get_te_params()
        start_frame = _safe_int(self._te_start_frame.text(), 0)
        if self._te_vectorized.isChecked():
            frame = create_time_encoded_frame_vectorized(
                motion,
                colormap=te_params['colormap'],
                window=te_params['window'],
                start_time=start_frame,
                scale_factor=te_params['scale_factor'],
                offset=te_params['offset'],
                light_background=te_params['light_background'],
            )
        else:
            frame = create_time_encoded_frame(
                motion,
                colormap=te_params['colormap'],
                window=te_params['window'],
                start_time=start_frame,
                scale_factor=te_params['scale_factor'],
                offset=te_params['offset'],
                light_background=te_params['light_background'],
            )
        _scheduler.schedule(lambda f=frame: self._save_frame_dialog(f))

    def _save_frame_dialog(self, frame):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Time Encoded Frame", "",
            "PNG Files (*.png);;JPEG Files (*.jpg)")
        if not path:
            return
        import cv2
        cv2.imwrite(path, frame)
        QMessageBox.information(self, "Success", f"Saved frame to {path}")

    def _do_save_video(self):
        if self._motion is None:
            QMessageBox.critical(self, "Error", "Please preview time encoding first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Time Encoded Video", "",
            "MP4 Files (*.mp4);;AVI Files (*.avi)")
        if not path:
            return
        if self._te_parallel.isChecked():
            trails = wts.create_time_encoded_array_parallel(
                self._motion,
                colormap=self._te_params['colormap'],
                window=self._te_params['window'],
                scale_factor=self._te_params['scale_factor'],
                offset=self._te_params['offset'],
                light_background=self._te_params['light_background'],
            )
        else:
            trails = wts.create_time_encoded_array(
                self._motion,
                colormap=self._te_params['colormap'],
                window=self._te_params['window'],
                scale_factor=self._te_params['scale_factor'],
                offset=self._te_params['offset'],
                light_background=self._te_params['light_background'],
            )
        if path.endswith('.avi'):
            wts.write_avi(trails, path)
        else:
            wts.write_mp4(trails, path)
        QMessageBox.information(self, "Success", f"Saved video to {path}")

    def _get_calibration(self, px_per_mm_entry, fps_entry):
        px = _safe_float(px_per_mm_entry.text(), None)
        fps = _safe_float(fps_entry.text(), None)
        if px is not None and fps is not None:
            return wts.Calibration(pixels_per_mm=px, frames_per_second=fps)
        return None

    # --- Computer Assisted Tab ---
    def setup_assisted_tab(self):
        layout = QVBoxLayout(self.tab_assisted)

        desc = QLabel(
            "Manually mark worms by double-clicking on the video overlay. "
            "Hold Shift to add another marker to the same worm.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc)

        cal_frame = CollapsibleFrame("Calibration (for physical units)")
        self._assisted_px_per_mm = cal_frame.add_label_entry("Pixels per mm:", 0)
        self._assisted_fps = cal_frame.add_label_entry("Frames per second:", 1)
        layout.addWidget(cal_frame)

        self._assisted_result = QLabel("")
        self._assisted_result.setFont(QFont("Arial", 13, QFont.Bold))
        self._assisted_result.setStyleSheet("color: blue;")
        self._assisted_result.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._assisted_result)

        self._assisted_save_btn = QPushButton("Save Markers CSV")
        self._assisted_save_btn.setEnabled(False)
        self._assisted_save_btn.clicked.connect(self._save_count_markers)
        layout.addWidget(self._assisted_save_btn)

        btn_start = QPushButton("Start Count Assist")
        btn_start.clicked.connect(
            lambda: self.measure_action_wrapper(
                self._do_count_assist, "Preparing Count Assist..."))
        layout.addWidget(btn_start)

        layout.addStretch()

    def _do_count_assist(self):
        video = self._get_video()
        filename = os.path.basename(self.video_path)
        cal = self._get_calibration(self._assisted_px_per_mm, self._assisted_fps)

        def run():
            df = wts.count_assist(video, window_name=filename, calibration=cal)
            self._last_count_assist_df = df
            n = len(df)
            self._assisted_result.setText(f"Manual Count: {n}")
            self._assisted_save_btn.setEnabled(n > 0)

        _scheduler.schedule(run)

    def _save_count_markers(self):
        if self._last_count_assist_df is None or self._last_count_assist_df.empty:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Marker Coordinates", "",
            "CSV Files (*.csv)")
        if save_path:
            self._last_count_assist_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Success", f"Saved to {save_path}")

    # --- Computer Automated Tab ---
    def setup_automated_tab(self):
        layout = QVBoxLayout(self.tab_auto)

        desc = QLabel("Automatically count worms using motion detection.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc)

        # --- count_simple section ---
        simple_grp = QGroupBox("Count Simple (trail distances)")
        sg_layout = QVBoxLayout(simple_grp)

        simple_main = CollapsibleFrame("Parameters")
        self._simple_mt = simple_main.add_label_entry("Motion Threshold:", 0)
        self._simple_mt.setText("1.5")
        self._simple_dr = simple_main.add_label_entry("Dilation Radius:", 1)
        self._simple_dr.setText("2")
        self._simple_mr = simple_main.add_label_entry("Mask Radius:", 2)
        self._simple_mr.setText("375")
        self._simple_detail = simple_main.add_checkbutton(
            "Return details (distances / areas)", 3, default=True)
        sg_layout.addWidget(simple_main)

        cal_frame = CollapsibleFrame("Calibration (for physical units)")
        self._auto_px_per_mm = cal_frame.add_label_entry("Pixels per mm:", 0)
        self._auto_fps = cal_frame.add_label_entry("Frames per second:", 1)
        sg_layout.addWidget(cal_frame)

        btn_simple = QPushButton("Run Count Simple")
        btn_simple.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_count_simple, "Counting simple..."))
        sg_layout.addWidget(btn_simple)
        layout.addWidget(simple_grp)

        # --- count_video section ---
        video_grp = QGroupBox("Count Video (roaming / quiescent)")
        vg_layout = QVBoxLayout(video_grp)

        video_main = CollapsibleFrame("Parameters")
        self.count_min = video_main.add_label_entry("Min Worm Area:", 0)
        self.count_min.setText("20")
        self.count_max = video_main.add_label_entry("Max Worm Area:", 1)
        self.count_max.setText("300")
        vg_layout.addWidget(video_main)

        adv = CollapsibleFrame("Advanced Parameters")
        self._count_mwl = adv.add_label_entry("Max Worm Length:", 0)
        self._count_mwl.setText("30")
        self._count_wks = adv.add_label_entry("Worm Kernel Size:", 1)
        self._count_wks.setText("11")
        self._count_wt = adv.add_label_entry("Worm Thresh:", 2)
        self._count_wt.setText("5")
        self._count_mt = adv.add_label_entry("Motion Thresh (blank=auto):", 3)
        self._count_smt = adv.add_label_entry("Strict Motion Thresh (blank=auto):", 4)
        self._count_smd = adv.add_label_entry("Strict Motion Dilation:", 5)
        self._count_smd.setText("1")
        self._count_sd = adv.add_label_entry("Stationary Dilation:", 6)
        self._count_sd.setText("1")
        self._count_mr = adv.add_label_entry("Mask Radius:", 7)
        self._count_mr.setText("375")
        self._count_vis = adv.add_checkbutton("Return Visualization", 8)
        vg_layout.addWidget(adv)

        btn_count = QPushButton("Run Count Video")
        btn_count.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_count, "Counting worms..."))
        vg_layout.addWidget(btn_count)
        layout.addWidget(video_grp)

        self.auto_progress = ProgressLabel()
        layout.addWidget(self.auto_progress)

        self.auto_result = QLabel("")
        self.auto_result.setFont(QFont("Arial", 13, QFont.Bold))
        self.auto_result.setStyleSheet("color: blue;")
        self.auto_result.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.auto_result)

        layout.addStretch()

    def _get_count_params(self):
        return {
            'min_worm_area': _safe_int(self.count_min.text(), 20),
            'max_worm_area': _safe_int(self.count_max.text(), 300),
            'max_worm_length': _safe_int(self._count_mwl.text(), 30),
            'worm_kernel_size': _safe_int(self._count_wks.text(), 11),
            'worm_thresh': _safe_int(self._count_wt.text(), 5),
            'motion_thresh': _safe_int(self._count_mt.text(), None),
            'strict_motion_thresh': _safe_int(self._count_smt.text(), None),
            'strict_motion_dilation': _safe_int(self._count_smd.text(), 1),
            'stationary_dilation': _safe_int(self._count_sd.text(), 1),
            'mask_radius': _safe_int(self._count_mr.text(), 375),
            'return_vis': self._count_vis.isChecked(),
        }

    def _do_count(self):
        video = self._get_video()
        n_roaming, n_stationary, vis = wts.count_video(video, **self._get_count_params())
        _scheduler.schedule(lambda: self.auto_result.setText(
            f"Roaming: {n_roaming}   Stationary: {n_stationary}"))
        if self._count_vis.isChecked():
            _scheduler.schedule(lambda: wts.show_video_array(vis))

    def _do_count_simple(self):
        video = self._get_video()
        cal = self._get_calibration(self._auto_px_per_mm, self._auto_fps)
        detail = self._simple_detail.isChecked()
        df = wts.count_simple(
            video,
            motion_thresh=_safe_float(self._simple_mt.text(), 1.5),
            dilation_radius=_safe_int(self._simple_dr.text(), 2),
            mask_radius=_safe_int(self._simple_mr.text(), 375),
            return_detail=detail,
            calibration=cal,
        )
        if detail:
            n = len(df)
            _scheduler.schedule(lambda d=df, n=n: self._auto_save_simple(d, n))
        else:
            _scheduler.schedule(lambda: self.auto_result.setText(f"Worms detected: {df}"))

    def _auto_save_simple(self, df, n):
        self.auto_result.setText(f"Worm trails detected: {n}")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Count Simple Results", "",
            "CSV Files (*.csv)")
        if save_path:
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Success", f"Saved to {save_path}")

    # --- Chemotaxis Tab ---
    def setup_chemotaxis_tab(self):
        layout = QVBoxLayout(self.tab_chemotaxis)

        desc = QLabel(
            "Measure trajectory, speed, and relative angle towards a bait spot over time windows.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc)

        prep = CollapsibleFrame("Preprocessing")
        self._chem_motion_method = prep.add_combobox(
            "Motion Method:", ["subtract_average", "linear_model_residuals"], 0,
            default="subtract_average")
        self._chem_vig_k = prep.add_label_entry("Vignetting Kernel (blank=auto):", 1)
        self._chem_vig_m = prep.add_checkbutton("Vignetting: Median Blur", 2)
        self._chem_sub_s = prep.add_label_entry("Subtract Avg Start:", 3)
        self._chem_sub_s.setText("0")
        self._chem_sub_e = prep.add_label_entry("Subtract Avg End (blank=last):", 4)
        self._chem_sub_a = prep.add_checkbutton("Subtract: Absolute Diff", 5, default=True)
        self._chem_sub_p = prep.add_checkbutton("Subtract: Use Projection", 6)
        self._chem_sub_l = prep.add_checkbutton("Subtract: Light Background", 7, default=True)
        self._chem_thresh = prep.add_label_entry("Threshold Value:", 8)
        self._chem_thresh.setText("30")
        layout.addWidget(prep)

        anal_grp = QGroupBox("Analysis Parameters")
        anal_grid = QGridLayout(anal_grp)
        anal_grid.addWidget(QLabel("Time Window (frames):"), 0, 0)
        self.chemo_window = QLineEdit("10")
        self.chemo_window.setMaximumWidth(80)
        anal_grid.addWidget(self.chemo_window, 0, 1)
        anal_grid.addWidget(QLabel("Interval (frames):"), 1, 0)
        self.chemo_int = QLineEdit("60")
        self.chemo_int.setMaximumWidth(80)
        anal_grid.addWidget(self.chemo_int, 1, 1)
        anal_grid.addWidget(QLabel("Min Size (px):"), 2, 0)
        self.chemo_min = QLineEdit("10")
        self.chemo_min.setMaximumWidth(80)
        anal_grid.addWidget(self.chemo_min, 2, 1)
        anal_grid.addWidget(QLabel("Max Size (px):"), 3, 0)
        self.chemo_max = QLineEdit("1000")
        self.chemo_max.setMaximumWidth(80)
        anal_grid.addWidget(self.chemo_max, 3, 1)
        layout.addWidget(anal_grp)

        bait_grp = QGroupBox("Bait Spot (optional)")
        bait_grid = QGridLayout(bait_grp)
        bait_grid.addWidget(QLabel("X:"), 0, 0)
        self.chemo_bait_x = QLineEdit()
        self.chemo_bait_x.setMaximumWidth(80)
        bait_grid.addWidget(self.chemo_bait_x, 0, 1)
        bait_grid.addWidget(QLabel("Y:"), 0, 2)
        self.chemo_bait_y = QLineEdit()
        self.chemo_bait_y.setMaximumWidth(80)
        bait_grid.addWidget(self.chemo_bait_y, 0, 3)
        layout.addWidget(bait_grp)

        chemo_cal = CollapsibleFrame("Calibration (for physical units)")
        self._chemo_px_per_mm = chemo_cal.add_label_entry("Pixels per mm:", 0)
        self._chemo_fps = chemo_cal.add_label_entry("Frames per second:", 1)
        layout.addWidget(chemo_cal)

        self._chemo_parallel = QCheckBox("Use Parallel")
        self._chemo_parallel.setChecked(False)
        layout.addWidget(self._chemo_parallel)

        self.chemo_progress = ProgressLabel()
        layout.addWidget(self.chemo_progress)

        btn_chemo = QPushButton("Measure Chemotaxis & Save CSV")
        btn_chemo.clicked.connect(
            lambda: self.measure_action_wrapper(self._do_chemo, "Measuring chemotaxis..."))
        layout.addWidget(btn_chemo)

        layout.addStretch()

    def _get_chemo_prep_params(self):
        vig_kernel = _safe_int(self._chem_vig_k.text(), None)
        sub_end_raw = self._chem_sub_e.text()
        sub_end = _safe_int(sub_end_raw, -1)
        if sub_end_raw == "":
            sub_end = -1
        return {
            'motion_method': self._chem_motion_method.currentText(),
            'vig': {'kernel_size': vig_kernel, 'use_median_blur': self._chem_vig_m.isChecked()},
            'sub': {
                'average_start': _safe_int(self._chem_sub_s.text(), 0),
                'average_end': sub_end,
                'use_absolute_difference': self._chem_sub_a.isChecked(),
                'use_projection': self._chem_sub_p.isChecked(),
                'light_background': self._chem_sub_l.isChecked(),
            },
            'threshold': _safe_int(self._chem_thresh.text(), 30),
        }

    def _do_chemo(self):
        prep = self._get_chemo_prep_params()
        corrected = self._ensure_corrected(prep['vig'])
        motion = self._ensure_motion(corrected, prep['motion_method'], prep['sub']).copy()
        thresh_val = prep['threshold']
        motion[motion > thresh_val] = 255
        motion[motion <= thresh_val] = 0

        bx = self.chemo_bait_x.text().strip()
        by = self.chemo_bait_y.text().strip()
        test_spot = None
        if bx and by:
            try:
                test_spot = (float(by), float(bx))
            except ValueError:
                _scheduler.schedule(lambda: QMessageBox.warning(
                    self, "Invalid Input", "Bait spot coordinates must be numeric."))
                return

        cal = self._get_calibration(self._chemo_px_per_mm, self._chemo_fps)
        if self._chemo_parallel.isChecked():
            df = wts.measure_chemotaxis_parallel(
                motion,
                time_window=_safe_int(self.chemo_window.text(), 10),
                interval=_safe_int(self.chemo_int.text(), 60),
                minimum_size=_safe_int(self.chemo_min.text(), 10),
                maximum_size=_safe_int(self.chemo_max.text(), 1000),
                test_spot=test_spot,
                calibration=cal,
            )
        else:
            df = wts.measure_chemotaxis(
                motion,
                time_window=_safe_int(self.chemo_window.text(), 10),
                interval=_safe_int(self.chemo_int.text(), 60),
                minimum_size=_safe_int(self.chemo_min.text(), 10),
                maximum_size=_safe_int(self.chemo_max.text(), 1000),
                test_spot=test_spot,
                calibration=cal,
            )

        def request_save():
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Chemotaxis Data", "",
                "CSV Files (*.csv)")
            if save_path:
                df.to_csv(save_path, index=False)
                QMessageBox.information(self, "Success", f"Saved to {save_path}")

        _scheduler.schedule(request_save)


def main():
    """Launch the Wormtrails GUI application."""
    app = QApplication(sys.argv)
    window = WormtrailsGUI()
    window.show()
    sys.exit(app.exec())