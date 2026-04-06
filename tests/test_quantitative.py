import unittest
import numpy as np
import pandas as pd
import os
from wormtrails.src.quantitative import count_video, measure_chemotaxis, track_and_label_worms
from wormtrails.src.file_io import read_video_file

class TestQuantitative(unittest.TestCase):
    def setUp(self):
        self.video = np.ones((10, 200, 200), dtype=np.uint8) * 200
        # draw a simple moving "worm"
        for i in range(10):
            # object moves from (30,30) to (30+i*2, 30+i*2)
            self.video[i, 30+i*2:40+i*2, 30+i*2:40+i*2] = 50
            
        self.binary_video = np.zeros((10, 200, 200), dtype=np.uint8)
        for i in range(10):
            self.binary_video[i, 30+i*2:40+i*2, 30+i*2:40+i*2] = 255

    def get_test_video(self, filename):
        path = os.path.join(os.path.dirname(__file__), filename)
        return read_video_file(path)

    def test_count_video(self):
        # We test with detailed_output=False
        count = count_video(self.video, min_size=5, max_size=500, detailed_output=False)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 0)

    def test_count_young_adults(self):
        video = self.get_test_video("lifespan_plate.avi")
        count = count_video(video, min_size=10, max_size=300, detailed_output=False, corrected_thresh=170, plate_edge_size=150)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 80)
        self.assertLessEqual(count, 100)

    def test_track_young_adults(self):
        video = self.get_test_video("lifespan_plate.avi")
        # Get binary array for tracking: stationary (128) and travelling (255) are both 255
        _, worms = count_video(video, min_size=10, max_size=300, detailed_output=True, corrected_thresh=170, plate_edge_size=150)
        binary = (worms >= 128).astype(np.uint8) * 255
        labeled = track_and_label_worms(binary, persistence=5)
        n_unique = len(np.unique(labeled)) - 1
        self.assertGreaterEqual(n_unique, 80)
        self.assertLessEqual(n_unique, 100)

    def test_count_old_adults(self):
        video = self.get_test_video("dying_plate.avi")
        count = count_video(video, min_size=10, max_size=300, detailed_output=False, corrected_thresh=170, plate_edge_size=150)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 35)
        self.assertLessEqual(count, 45)

    def test_track_old_adults(self):
        video = self.get_test_video("dying_plate.avi")
        _, worms = count_video(video, min_size=10, max_size=300, detailed_output=True, corrected_thresh=170, plate_edge_size=150)
        binary = (worms >= 128).astype(np.uint8) * 255
        labeled = track_and_label_worms(binary, persistence=5)
        n_unique = len(np.unique(labeled)) - 1
        self.assertGreaterEqual(n_unique, 30)
        self.assertLessEqual(n_unique, 50)

    def test_measure_chemotaxis(self):
        df = measure_chemotaxis(self.binary_video, time_window=5, interval=2, minimum_size=5, maximum_size=500, test_spot=(100,100))
        self.assertTrue(isinstance(df, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
