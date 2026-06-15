import sys
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QWidget, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QPoint, QEvent
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QKeyEvent,
    QMouseEvent, QWheelEvent, QCursor
)
from .processing import create_time_encoded_frame, fit_pixel_linear_model

__all__ = [
    'show_video_array',
    'show_frame',
    'show_time_encoding',
    'count_assist',
    'select_bait_spot',
]


def _ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _numpy_to_qpixmap(arr):
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        buf = arr.tobytes()
        img = QImage(buf, w, h, w, QImage.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w = arr.shape[:2]
        rgb = np.ascontiguousarray(arr[:, :, ::-1])
        buf = rgb.tobytes()
        img = QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    return QPixmap.fromImage(img.copy())


def _render_zoomed(orig_pm, label_size, zoom, pan_dx, pan_dy):
    """Paint *orig_pm* into a *label_size*-sized pixmap with zoom and pan.

    Returns (result_pixmap, draw_x, draw_y, display_w, display_h)
    where (draw_x, draw_y) is the top-left corner of the image content
    within the result, and (display_w, display_h) is its rendered size.
    """
    if zoom <= 0:
        zoom = 1.0

    fit = orig_pm.size().scaled(label_size, Qt.KeepAspectRatio)
    dw = max(1, int(fit.width() * zoom))
    dh = max(1, int(fit.height() * zoom))
    scaled = orig_pm.scaled(dw, dh, Qt.KeepAspectRatio, Qt.FastTransformation)
    dw = scaled.width()
    dh = scaled.height()

    max_dx = max(0, (dw - label_size.width()) // 2)
    max_dy = max(0, (dh - label_size.height()) // 2)
    pan_dx = max(-max_dx, min(max_dx, pan_dx))
    pan_dy = max(-max_dy, min(max_dy, pan_dy))

    draw_x = (label_size.width() - dw) // 2 + pan_dx
    draw_y = (label_size.height() - dh) // 2 + pan_dy

    result = QPixmap(label_size)
    result.fill(Qt.black)
    p = QPainter(result)
    p.drawPixmap(draw_x, draw_y, scaled)
    p.end()

    return result, draw_x, draw_y, dw, dh


class _ZoomableLabel(QLabel):
    """A QLabel that forwards wheel events to a callback."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel_callback = None

    def set_wheel_callback(self, callback):
        self._wheel_callback = callback

    def wheelEvent(self, event):
        if self._wheel_callback is not None:
            self._wheel_callback(event)
            event.accept()
        else:
            super().wheelEvent(event)


class _ZoomPanMixin:
    """Adds scroll-to-zoom and drag-to-pan to an image_label-based dialog.

    The ``image_label`` must be a :class:`_ZoomableLabel` with its wheel
    callback connected via :meth:`_connect_wheel`.
    """

    def _init_zoom_pan(self):
        self._zoom = 1.0
        self._pan_dx = 0
        self._pan_dy = 0
        self._dragging = False
        self._drag_start = QPoint()
        self._drag_pan_start = (0, 0)

    def _connect_wheel(self):
        self.image_label.set_wheel_callback(self._handle_wheel)

    def _paint_base(self, orig_pm, label_size):
        """Apply zoom/pan and return (pixmap, draw_x, draw_y, dw, dh)."""
        pm, dx, dy, dw, dh = _render_zoomed(
            orig_pm, label_size, self._zoom, self._pan_dx, self._pan_dy)
        self._tr = (dx, dy, dw, dh, orig_pm.width(), orig_pm.height())
        return pm, dx, dy, dw, dh

    def _label_to_image(self, lx, ly):
        dx, dy, dw, dh, ow, oh = self._tr
        return (lx - dx) * ow / dw, (ly - dy) * oh / dh

    def _image_to_label(self, ix, iy):
        dx, dy, dw, dh, ow, oh = self._tr
        return dx + ix * dw / ow, dy + iy * dh / oh

    def _cap_pan(self):
        if self._original_pixmap is None:
            return
        label_size = self.image_label.size()
        fit = self._original_pixmap.size().scaled(label_size, Qt.KeepAspectRatio)
        dw = max(1, int(fit.width() * self._zoom))
        dh = max(1, int(fit.height() * self._zoom))
        max_dx = max(0, (dw - label_size.width()) // 2)
        max_dy = max(0, (dh - label_size.height()) // 2)
        self._pan_dx = max(-max_dx, min(max_dx, self._pan_dx))
        self._pan_dy = max(-max_dy, min(max_dy, self._pan_dy))

    def _handle_wheel(self, event):
        if self._original_pixmap is None:
            return

        old_zoom = self._zoom
        factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self._zoom = max(0.1, min(10.0, self._zoom * factor))

        ls = self.image_label.size()
        fit = self._original_pixmap.size().scaled(ls, Qt.KeepAspectRatio)
        old_dw = max(1, int(fit.width() * old_zoom))
        old_dh = max(1, int(fit.height() * old_zoom))
        new_dw = max(1, int(fit.width() * self._zoom))
        new_dh = max(1, int(fit.height() * self._zoom))

        cx, cy = event.position().x(), event.position().y()
        old_dx = (ls.width() - old_dw) // 2 + self._pan_dx
        old_dy = (ls.height() - old_dh) // 2 + self._pan_dy
        img_x = (cx - old_dx) * self._original_pixmap.width() / old_dw
        img_y = (cy - old_dy) * self._original_pixmap.height() / old_dh

        new_dx = cx - img_x * new_dw / self._original_pixmap.width()
        new_dy = cy - img_y * new_dh / self._original_pixmap.height()

        self._pan_dx = new_dx - (ls.width() - new_dw) // 2
        self._pan_dy = new_dy - (ls.height() - new_dh) // 2
        self._cap_pan()
        self._refresh_display()

    def _try_start_drag(self, event):
        if (event.button() == Qt.LeftButton and self._zoom > 1.0
                and self._original_pixmap is not None):
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_pan_start = (self._pan_dx, self._pan_dy)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return True
        return False

    def _try_drag_move(self, event):
        if self._dragging:
            delta = event.position().toPoint() - self._drag_start
            self._pan_dx = self._drag_pan_start[0] + delta.x()
            self._pan_dy = self._drag_pan_start[1] + delta.y()
            self._cap_pan()
            self._refresh_display()
            return True
        return False

    def _try_end_drag(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            return True
        return False


class _VideoPlayerDialog(QDialog, _ZoomPanMixin):
    def __init__(self, video_array, window_title="Video", show_slider=True,
                 frame_callback=None, parent=None):
        super().__init__(parent)
        self.video_array = video_array
        self.num_frames = video_array.shape[0]
        self.current_idx = 0
        self.playing = False
        self.frame_callback = frame_callback

        self.setWindowTitle(window_title)
        self.setMinimumSize(400, 300)
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        self.image_label = _ZoomableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

        if show_slider:
            slider_layout = QHBoxLayout()
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.num_frames - 1)
            self.slider.valueChanged.connect(self._on_slider)
            slider_layout.addWidget(self.slider)

            self.play_btn = QPushButton("Play")
            self.play_btn.clicked.connect(self._toggle_play)
            slider_layout.addWidget(self.play_btn)

            self.frame_label = QLabel(f"0 / {self.num_frames - 1}")
            slider_layout.addWidget(self.frame_label)
            layout.addLayout(slider_layout)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._advance_frame)

        self._init_zoom_pan()
        self._connect_wheel()
        self._original_pixmap = None
        self._tr = (0, 0, 1, 1, 1, 1)
        self._show_frame(0)

    def _refresh_display(self):
        if self._original_pixmap is None:
            return
        pm, _, _, _, _ = self._paint_base(self._original_pixmap, self.image_label.size())
        self.image_label.setPixmap(pm)

    def _show_frame(self, idx):
        if idx < 0 or idx >= self.num_frames:
            return
        self.current_idx = idx
        frame_data = self.video_array[idx]

        if self.frame_callback:
            frame_data = self.frame_callback(idx, frame_data)

        if frame_data.ndim == 2:
            frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)

        self._original_pixmap = _numpy_to_qpixmap(frame_data)
        self._cap_pan()
        self._refresh_display()

    def _on_slider(self, value):
        self._show_frame(value)
        if self.frame_label:
            self.frame_label.setText(f"{value} / {self.num_frames - 1}")

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _advance_frame(self):
        nxt = self.current_idx + 1
        if nxt >= self.num_frames:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
            return
        self.slider.setValue(nxt)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._cap_pan()
        self._refresh_display()

    def mousePressEvent(self, event):
        if not self._try_start_drag(event):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._try_drag_move(event):
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._try_end_drag(event):
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        elif event.key() == Qt.Key_Space:
            if hasattr(self, 'play_btn'):
                self._toggle_play()
        elif event.key() == Qt.Key_Left and hasattr(self, 'slider'):
            self.slider.setValue(max(0, self.current_idx - 1))
        elif event.key() == Qt.Key_Right and hasattr(self, 'slider'):
            self.slider.setValue(min(self.num_frames - 1, self.current_idx + 1))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def show_video_array(video_array, window_name="esc to exit"):
    num_frames = video_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return

    _ensure_qapp()
    dialog = _VideoPlayerDialog(video_array, window_title=window_name)
    dialog.exec()


def show_frame(frame, window_name="esc to exit"):
    _ensure_qapp()
    if frame.ndim == 2:
        h, w = frame.shape
        expected = 1
    elif frame.ndim == 3:
        h, w = frame.shape[:2]
        expected = frame.shape[2]
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    video = frame.reshape(1, h, w) if expected == 1 else frame.reshape(1, h, w, expected)
    dialog = _VideoPlayerDialog(video, window_title=window_name,
                                show_slider=False)
    dialog.exec()


def show_time_encoding(average_subtracted_array,
                       colormap=np.array([[0, 0, 0]]),
                       window=1, scale_factor=1, offset=0,
                       light_background=True, window_name="esc to exit"):
    num_frames = average_subtracted_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return

    _ensure_qapp()

    def te_callback(idx, frame_data):
        return create_time_encoded_frame(
            average_subtracted_array, colormap=colormap, window=window,
            start_time=idx, scale_factor=scale_factor, offset=offset,
            light_background=light_background
        )

    max_slider = max(0, num_frames - window)
    dialog = _VideoPlayerDialog(average_subtracted_array,
                                window_title=window_name,
                                frame_callback=te_callback)
    dialog.slider.setMaximum(max_slider)
    dialog.frame_label.setText(f"0 / {max_slider}")
    dialog.exec()


class _BaitSpotDialog(QDialog, _ZoomPanMixin):
    def __init__(self, frame, window_title="Select Bait Spot", parent=None):
        super().__init__(parent)
        self.selected_point = None
        self._frame = frame
        self.setWindowTitle(window_title)
        self.setMinimumSize(400, 300)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        instr = QLabel("Double-click on the bait spot. Esc or Q to cancel.")
        layout.addWidget(instr)

        self.image_label = _ZoomableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

        btn_layout = QHBoxLayout()
        self._coord_label = QLabel("")
        btn_layout.addWidget(self._coord_label)
        btn_layout.addStretch()
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(confirm_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._init_zoom_pan()
        self._connect_wheel()

        self._original_pixmap = _numpy_to_qpixmap(frame)
        self._cap_pan()
        self._refresh_display()

    def _refresh_display(self):
        if self._original_pixmap is None:
            return
        pm, _, _, _, _ = self._paint_base(self._original_pixmap, self.image_label.size())
        self.image_label.setPixmap(pm)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._original_pixmap is None:
            return

        label_pos = self.image_label.mapFrom(self, event.position().toPoint())
        img_x, img_y = self._label_to_image(label_pos.x(), label_pos.y())

        h, w = self._frame.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            self.selected_point = (int(img_x), int(img_y))
            self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.reject()
        else:
            super().keyPressEvent(event)


def select_bait_spot(frame, window_name="Select Bait Spot"):
    _ensure_qapp()
    dialog = _BaitSpotDialog(frame, window_title=window_name)
    if dialog.exec() == QDialog.Accepted:
        return dialog.selected_point
    return None


class _CountAssistDialog(QDialog, _ZoomPanMixin):
    def __init__(self, overlay_video, window_title, calibration, parent=None):
        super().__init__(parent)
        self.overlay_video = overlay_video
        self.num_frames = len(overlay_video)
        self.calibration = calibration
        self.current_idx = 1
        self.markers = []
        self.needs_redraw = True

        self.setWindowTitle(window_title)
        self.setMinimumSize(400, 300)
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        self.image_label = _ZoomableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMouseTracking(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(overlay_video) - 1)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_slider)
        slider_layout.addWidget(self.slider)

        self.frame_label = QLabel(f"1 / {len(overlay_video) - 1}")
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)

        self._init_zoom_pan()
        self._connect_wheel()
        self._original_pixmap = None
        self._tr = (0, 0, 1, 1, 1, 1)
        self._render_frame()

    def _overlay_frame(self, idx):
        return max(0, idx - 2)

    def _next_worm_id(self):
        return max((m['worm_id'] for m in self.markers), default=0) + 1

    def _add_marker(self, x, y, worm_id):
        self.markers.append({
            'worm_id': worm_id,
            'x': x,
            'y': y,
            'frame': self._overlay_frame(self.current_idx),
        })
        distinct = len({m['worm_id'] for m in self.markers})
        print(f"Markers: {len(self.markers)}, Worms: {distinct}", end='\r')
        self._render_frame()

    def _render_frame(self):
        idx = self.current_idx
        frame = self.overlay_video[idx]
        if frame.dtype in (np.float32, np.float64):
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self._original_pixmap = _numpy_to_qpixmap(frame)
        self._cap_pan()
        pm, _, _, _, _ = self._paint_base(self._original_pixmap,
                                          self.image_label.size())

        painter = QPainter(pm)
        pen = QPen(QColor(255, 0, 0), 1)
        painter.setPen(pen)

        worm_groups = {}
        for m in self.markers:
            wid = m['worm_id']
            worm_groups.setdefault(wid, []).append((m['x'], m['y']))

        for pts in worm_groups.values():
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    lx1, ly1 = self._image_to_label(pts[i][0], pts[i][1])
                    lx2, ly2 = self._image_to_label(pts[i + 1][0], pts[i + 1][1])
                    painter.drawLine(int(round(lx1)), int(round(ly1)),
                                     int(round(lx2)), int(round(ly2)))

        font = QFont("Arial", 16)
        painter.setFont(font)
        for m in self.markers:
            lx, ly = self._image_to_label(m['x'], m['y'])
            rx, ry = int(round(lx)), int(round(ly))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(rx - 3, ry - 3, 6, 6)
            painter.drawText(rx + 4, ry - 4, str(m['worm_id']))

        painter.end()

        self.image_label.setPixmap(pm)

    def _refresh_display(self):
        self._render_frame()

    def _on_slider(self, value):
        self.current_idx = value
        self._render_frame()
        self.frame_label.setText(f"{value} / {self.num_frames - 1}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._original_pixmap is not None:
            self._cap_pan()
            self._render_frame()

    def mousePressEvent(self, event):
        if not self._try_start_drag(event):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._try_drag_move(event):
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._try_end_drag(event):
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return

        if self._original_pixmap is None:
            return

        label_pos = self.image_label.mapFrom(self, event.position().toPoint())
        img_x, img_y = self._label_to_image(label_pos.x(), label_pos.y())

        ow, oh = self._original_pixmap.width(), self._original_pixmap.height()
        if img_x < 0 or img_y < 0 or img_x >= ow or img_y >= oh:
            return

        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.ShiftModifier) and self.markers:
            worm_id = self.markers[-1]['worm_id']
        else:
            worm_id = self._next_worm_id()

        self._add_marker(int(img_x), int(img_y), worm_id)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if self.markers:
                self.markers.pop()
                distinct = len({m['worm_id'] for m in self.markers}) if self.markers else 0
                print(f"Markers: {len(self.markers)}, Worms: {distinct}", end='\r')
                self._render_frame()
        elif event.key() == Qt.Key_Left:
            self.slider.setValue(max(0, self.current_idx - 1))
        elif event.key() == Qt.Key_Right:
            self.slider.setValue(min(self.num_frames - 1, self.current_idx + 1))
        else:
            super().keyPressEvent(event)

    def get_dataframe(self):
        df = pd.DataFrame(self.markers)
        if df.empty:
            return pd.DataFrame(columns=['worm_id', 'x', 'y', 'frame'])

        if self.calibration is not None:
            df['x_mm'] = self.calibration.distance_mm(df['x'].values)
            df['y_mm'] = self.calibration.distance_mm(df['y'].values)
            df['time_s'] = df['frame'].values / self.calibration.frames_per_second

        return df


def count_assist(video_array, window_name="count assist", calibration=None):
    num_frames = video_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return pd.DataFrame(columns=['worm_id', 'x', 'y', 'frame'])

    residuals, _, _ = fit_pixel_linear_model(video_array)
    motion_proj = np.mean(residuals ** 2, axis=0)
    motion_proj[motion_proj < 1] = 1
    motion_proj = np.log2(motion_proj.astype(np.float64))
    max_motion = np.max(motion_proj)
    if max_motion == 0:
        raise ValueError(
            "All frames are identical — no motion detected. "
            "Cannot compute motion overlay for count_assist."
        )
    motion_proj *= 255 / max_motion
    motion_proj = np.clip(motion_proj, 0, 255).astype(np.uint8)

    time_derivative = video_array.copy().astype(np.float16)[1:] - video_array.copy().astype(np.float16)[:-1]
    time_derivative = np.abs(time_derivative)
    time_derivative[time_derivative < 1] = 1
    time_derivative = np.log2(time_derivative)
    max_motion = np.max(time_derivative)
    if max_motion == 0:
        raise ValueError(
            "All frames are identical — no motion detected. "
            "Cannot compute motion overlay for count_assist."
        )
    time_derivative *= 255 / max_motion
    time_derivative = np.clip(time_derivative, 0, 255).astype(np.uint8)

    overlay_video = []
    overlay_video.append(video_array[0] // 2)
    overlay_video.append(np.mean(video_array, axis=0) // 2 + motion_proj // 2)
    for t in range(num_frames - 1):
        overlay_video.append(
            np.clip((video_array[t] // 2) + (time_derivative[t] // 2), 0, 255).astype(np.uint8)
        )

    _ensure_qapp()
    dialog = _CountAssistDialog(overlay_video, window_title=window_name,
                                calibration=calibration)
    dialog.exec()
    return dialog.get_dataframe()