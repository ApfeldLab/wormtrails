import unittest
import numpy as np
import tempfile
import os
from wormtrails.file_io import read_video_file, write_mp4, write_avi


class TestFileIO(unittest.TestCase):
    def test_read_nonexistent_file_raises(self):
        with self.assertRaises(ValueError):
            read_video_file("/nonexistent/path/video.avi")

    def test_write_mp4_grayscale(self):
        video = np.ones((5, 100, 100), dtype=np.uint8) * 128
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            write_mp4(video, path, fps=10)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
        finally:
            os.unlink(path)

    def test_write_mp4_color(self):
        video = np.ones((5, 100, 100, 3), dtype=np.uint8) * 128
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            write_mp4(video, path, fps=10)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
        finally:
            os.unlink(path)

    def test_write_avi_grayscale(self):
        video = np.ones((5, 100, 100), dtype=np.uint8) * 128
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as f:
            path = f.name
        try:
            write_avi(video, path, fps=10)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
        finally:
            os.unlink(path)

    def test_write_avi_color(self):
        video = np.ones((5, 100, 100, 3), dtype=np.uint8) * 128
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as f:
            path = f.name
        try:
            write_avi(video, path, fps=10)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
        finally:
            os.unlink(path)

    def test_read_video_file_from_written_mp4(self):
        video = np.ones((10, 100, 100), dtype=np.uint8) * 200
        video[3:7, 40:60, 40:60] = 50
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        try:
            write_mp4(video, path, fps=10)
            read_back = read_video_file(path)
            self.assertEqual(read_back.ndim, 3)
            self.assertEqual(read_back.dtype, np.uint8)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
