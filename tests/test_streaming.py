import unittest
from unittest import mock
import numpy as np
import cv2
from wormtrails.streaming import (
    get_average_frame,
    get_motion,
    get_time_encoded_frame,
    measure_chemotaxis_streaming,
    show_motion,
    show_time_encoded_frame,
    create_time_encoded_array_streaming,
)
from wormtrails.quantitative import Calibration


class MemoryCap:
    """In-memory stand-in for cv2.VideoCapture that reads frames from a Numpy
    array instead of a file on disk. Frames are returned as BGR, matching the
    format OpenCV's VideoCapture produces, so the streaming helpers behave
    identically without writing a temporary video file."""

    def __init__(self, frames):
        self._frames = np.asarray(frames)
        self._pos = 0

    def isOpened(self):
        return True

    def read(self):
        if self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frames.shape[2]
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frames.shape[1]
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self._pos
        return 0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
            return True
        return False

    def release(self):
        pass


def _make_frames(n_frames=8, height=60, width=80, moving_rows=(20, 45)):
    """Builds a small synthetic video in RAM where a bright bar moves down."""
    video = np.full((n_frames, height, width), 5, dtype=np.uint8)
    for t in range(n_frames):
        row = moving_rows[0] + t
        video[t, row:row + 6, 30:50] = 200
    return video


class TestStreamingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = _make_frames()
        cls.cap = MemoryCap(cls.frames)

    def test_get_average_frame_shape(self):
        avg = get_average_frame(self.cap)
        self.assertEqual(avg.ndim, 2)
        self.assertEqual(avg.shape, (60, 80))
        self.assertEqual(avg.dtype, np.float64)

    def test_get_motion_yields_frames(self):
        avg = get_average_frame(self.cap)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = list(get_motion(self.cap, reference_frame=avg, kernel_radius=5, window=4))
        self.assertEqual(len(frames), 4)
        for f in frames:
            self.assertEqual(f.shape, (60, 80))
            self.assertEqual(f.dtype, np.float64)

    def test_get_time_encoded_frame_shape(self):
        avg = get_average_frame(self.cap)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = get_time_encoded_frame(
            self.cap, reference_frame=avg, kernel_radius=5,
            start_frame=0, window=4, colormap=np.array([[0, 255, 0]]),
        )
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[-1], 3)
        self.assertEqual(frame.dtype, np.uint8)


class TestStreamingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = _make_frames()

    def test_nonexistent_file_raises(self):
        with self.assertRaises(ValueError):
            show_motion("/nonexistent/path/video.avi")
        with self.assertRaises(ValueError):
            show_time_encoded_frame("/nonexistent/path/video.avi", worm_length=5)
        with self.assertRaises(ValueError):
            measure_chemotaxis_streaming("/nonexistent/path/video.avi")
        with self.assertRaises(ValueError):
            create_time_encoded_array_streaming("/nonexistent/path/video.avi")

    def test_measure_chemotaxis_streaming_returns_df(self):
        # The streaming function opens the source via cv2.VideoCapture(path).
        # Substitute an in-memory cap so no temporary file is written.
        with mock.patch("cv2.VideoCapture", return_value=MemoryCap(self.frames)):
            df = measure_chemotaxis_streaming(
                "in-memory", thresh=5, worm_length=5,
                window=4, interval=3, minimum_size=10, maximum_size=1000,
                test_spot=(30, 40), calibration=Calibration(pixels_per_mm=10),
            )
        self.assertFalse(df.empty)
        for col in ("y", "x", "direction_y", "direction_x", "speed", "time",
                    "r", "theta", "relative_angle", "r_mm"):
            self.assertIn(col, df.columns)

    def test_measure_chemotaxis_streaming_accepts_mask_radius(self):
        # mask_radius=0 disables masking; a positive radius must be accepted
        # and applied without error.
        with mock.patch("cv2.VideoCapture", return_value=MemoryCap(self.frames)):
            masked = measure_chemotaxis_streaming(
                "in-memory", thresh=5, worm_length=5,
                window=4, interval=3, minimum_size=10, maximum_size=1000,
                mask_radius=30,
            )
        self.assertTrue(isinstance(masked, __import__("pandas").DataFrame))


if __name__ == "__main__":
    unittest.main()
