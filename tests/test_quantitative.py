import unittest
import numpy as np
import pandas as pd
import os
from wormtrails.quantitative import count_video, measure_chemotaxis
from wormtrails.file_io import read_video_file

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
        n_roaming, n_quiescent, vis = count_video(self.video, min_worm_area=5, max_worm_area=500)
        count = int(n_roaming + n_quiescent)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 0)

    def test_count_young_adults(self):
        video = self.get_test_video("lifespan_plate.avi")
        n_roaming, n_quiescent, vis = count_video(
            video,
            min_worm_area=20,
            max_worm_area=300,
            max_worm_length=30,
            worm_kernel_size=11,
            worm_thresh=5,
            motion_thresh=3,
            strict_motion_thresh=4,
            strict_motion_dilation=1,
            contrast_motion_correction_factor=50,
            edge_contrast_kernel_size=51,
            edge_contrast_thresh=4,
            mask_inclusion_kernel_size=31,
            edge_offset=0,
            return_vis=False
        )
        count = int(n_roaming + n_quiescent)
        print(count)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 95)
        self.assertLessEqual(count, 110)

    def test_count_old_adults(self):
        video = self.get_test_video("dying_plate.avi")
        n_roaming, n_quiescent, vis = count_video(
            video,
            min_worm_area=20,
            max_worm_area=300,
            max_worm_length=30,
            worm_kernel_size=11,
            worm_thresh=5,
            motion_thresh=3,
            strict_motion_thresh=4,
            strict_motion_dilation=1,
            contrast_motion_correction_factor=50,
            edge_contrast_kernel_size=51,
            edge_contrast_thresh=4,
            mask_inclusion_kernel_size=31,
            edge_offset=0,
            return_vis=False
        )
        count = int(n_roaming + n_quiescent)
        print(count)
        self.assertTrue(isinstance(count, int))
        self.assertGreaterEqual(count, 35)
        self.assertLessEqual(count, 45)

    def test_measure_chemotaxis(self):
        df = measure_chemotaxis(self.binary_video, time_window=5, interval=2, minimum_size=5, maximum_size=500, test_spot=(100,100))
        self.assertTrue(isinstance(df, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
