import unittest
import numpy as np
from wormtrails.src.processing import correct_vignetting, subtract_average, threshold_array, create_time_encoded_array

class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.video = np.ones((10, 200, 200), dtype=np.uint8) * 200
        # draw a simple moving "worm"
        for i in range(10):
            # object moves from (30,30) to (30+i*2, 30+i*2)
            self.video[i, 30+i*2:40+i*2, 30+i*2:40+i*2] = 50
    
    def test_correct_vignetting(self):
        result = correct_vignetting(self.video, kernel_size=11, inplace=False)
        self.assertEqual(result.shape, self.video.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_subtract_average(self):
        result = subtract_average(self.video, use_absolute_difference=True, inplace=False)
        self.assertEqual(result.shape, self.video.shape)
        # Check that there's motion in the array
        self.assertTrue(np.max(result) > 0)

    def test_threshold_array(self):
        result = threshold_array(self.video[0], threshold=100, dark_objects=True)
        self.assertEqual(result.shape, (200, 200))
        # Since it's all 100, and threshold is 80 & dark_objects=True (meaning things < 80 become True)
        self.assertTrue(np.max(result) == 255)

    def test_full_pipeline(self):
        corrected = correct_vignetting(self.video, kernel_size=11, inplace=False)
        motion = subtract_average(corrected, use_absolute_difference=True, inplace=False)
        time_encoded = create_time_encoded_array(motion, window=5)
        self.assertEqual(time_encoded.shape, (5, 200, 200, 3))
        self.assertTrue(np.max(time_encoded) > 0)

if __name__ == '__main__':
    unittest.main()
