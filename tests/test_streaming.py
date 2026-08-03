import unittest
import numpy as np
import cv2
import tempfile
import os
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


def _make_test_video(path, n_frames=8, height=60, width=80, moving_rows=(20, 45)):
    """Writes a small synthetic AVI where a bright bar moves down the screen."""
    if os.path.exists(path):
        return path
    from wormtrails.file_io import write_avi
    video = np.full((n_frames, height, width), 0, dtype=np.uint8)
    for t in range(n_frames):
        row = moving_rows[0] + t
        video[t, row:row + 6, 30:50] = 200
    write_avi(video, path, fps=10)
    return path


class TestStreamingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls.video_path = _make_test_video(os.path.join(cls._tmpdir, "test.avi"))
        cls.cap = cv2.VideoCapture(cls.video_path)

    @classmethod
    def tearDownClass(cls):
        cls.cap.release()
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

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
        cls._tmpdir = tempfile.mkdtemp()
        cls.video_path = _make_test_video(os.path.join(cls._tmpdir, "test.avi"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

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
        df = measure_chemotaxis_streaming(
            self.video_path, thresh=5, worm_length=5,
            window=4, interval=3, minimum_size=10, maximum_size=1000,
            test_spot=(30, 40), calibration=Calibration(pixels_per_mm=10),
        )
        self.assertFalse(df.empty)
        for col in ("y", "x", "direction_y", "direction_x", "speed", "time",
                    "r", "theta", "relative_angle", "r_mm"):
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
