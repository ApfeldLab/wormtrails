import unittest
import numpy as np
from wormtrails.processing import correct_vignetting, subtract_average, create_time_encoded_array, create_track_array

class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.video = np.ones((10, 200, 200), dtype=np.uint8) * 200
        # draw a simple moving "worm"
        for i in range(10):
            # object moves from (30,30) to (30+i*2, 30+i*2)
            self.video[i, 30+i*2:40+i*2, 30+i*2:40+i*2] = 50
    
    def test_correct_vignetting(self):
        result = correct_vignetting(self.video, kernel_size=11, inPlace=False)
        self.assertEqual(result.shape, self.video.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_subtract_average(self):
        result = subtract_average(self.video, use_absolute_difference=True, inPlace=False)
        self.assertEqual(result.shape, self.video.shape)
        # Check that there's motion in the array
        self.assertTrue(np.max(result) > 0)

    def test_threshold_array(self):
        result = create_track_array(self.video, window=1)
        self.assertEqual(result.shape, self.video.shape)

    def test_full_pipeline(self):
        corrected = correct_vignetting(self.video, kernel_size=11, inPlace=False)
        motion = subtract_average(corrected, use_absolute_difference=True, inPlace=False)
        time_encoded = create_time_encoded_array(motion, window=5)
        self.assertEqual(time_encoded.shape, (5, 200, 200, 3))
        self.assertTrue(np.max(time_encoded) > 0)

if __name__ == '__main__':
    unittest.main()
