import unittest
import numpy as np
from wormtrails.processing import correct_vignetting, subtract_average, create_time_encoded_array, create_track_array, fit_pixel_linear_model

class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.video = np.ones((10, 200, 200), dtype=np.uint8) * 200
        # draw a simple moving "worm"
        for i in range(10):
            # object moves from (30,30) to (30+i*2, 30+i*2)
            self.video[i, 30+i*2:40+i*2, 30+i*2:40+i*2] = 50
    
    def test_correct_vignetting(self):
        result = correct_vignetting(self.video, kernel_size=11, in_place=False)
        self.assertEqual(result.shape, self.video.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_subtract_average(self):
        result = subtract_average(self.video, use_absolute_difference=True, in_place=False)
        self.assertEqual(result.shape, self.video.shape)
        # Check that there's motion in the array
        self.assertTrue(np.max(result) > 0)

    def test_threshold_array(self):
        result = create_track_array(self.video, window=1)
        self.assertEqual(result.shape, self.video.shape)

    def test_full_pipeline(self):
        corrected = correct_vignetting(self.video, kernel_size=11, in_place=False)
        motion = subtract_average(corrected, use_absolute_difference=True, in_place=False)
        time_encoded = create_time_encoded_array(motion, window=5)
        self.assertEqual(time_encoded.shape, (5, 200, 200, 3))
        self.assertTrue(np.max(time_encoded) > 0)

    def test_fit_pixel_linear_model_output_shapes_and_dtypes(self):
        resid, slope, intercept = fit_pixel_linear_model(self.video)
        self.assertEqual(resid.shape, self.video.shape)
        self.assertEqual(slope.shape, (200, 200))
        self.assertEqual(intercept.shape, (200, 200))
        self.assertEqual(resid.dtype, np.float16)
        self.assertEqual(slope.dtype, np.float64)
        self.assertEqual(intercept.dtype, np.float64)

    def test_fit_pixel_linear_model_constant_video(self):
        const = np.full((10, 50, 50), 100, dtype=np.uint8)
        resid, slope, _ = fit_pixel_linear_model(const)
        self.assertAlmostEqual(np.max(np.abs(resid)), 0, places=5)
        self.assertAlmostEqual(np.max(np.abs(slope)), 0, places=5)

    def test_fit_pixel_linear_model_raises_on_1d(self):
        with self.assertRaises(ValueError):
            fit_pixel_linear_model(np.array([1, 2, 3], dtype=np.uint8))

    def test_fit_pixel_linear_model_raises_on_lt2_frames(self):
        with self.assertRaises(ValueError):
            fit_pixel_linear_model(np.array([[1]], dtype=np.uint8).reshape(1, 1, 1))

    def test_fit_pixel_linear_model_nonzero_motion(self):
        resid, _, _ = fit_pixel_linear_model(self.video)
        self.assertGreater(np.max(np.abs(resid)), 0)

    def test_time_encoded_array_backends_equivalent(self):
        motion = subtract_average(self.video, use_absolute_difference=True, in_place=False)
        seq = create_time_encoded_array(motion, window=3, mode='sequential')
        vec = create_time_encoded_array(motion, window=3, mode='vectorized')
        par = create_time_encoded_array(motion, window=3, mode='parallel')
        self.assertTrue(np.array_equal(seq, vec))
        self.assertTrue(np.array_equal(seq, par))

    def test_time_encoded_array_auto_routes_to_sequential(self):
        motion = subtract_average(self.video, use_absolute_difference=True, in_place=False)
        auto = create_time_encoded_array(motion, window=3, mode='auto')
        seq = create_time_encoded_array(motion, window=3, mode='sequential')
        self.assertTrue(np.array_equal(auto, seq))

    def test_time_encoded_array_path_requires_stream_mode(self):
        with self.assertRaises(ValueError):
            create_time_encoded_array("does-not-exist.avi", mode='sequential')

    def test_time_encoded_array_array_rejects_stream_mode(self):
        with self.assertRaises(ValueError):
            create_time_encoded_array(self.video, mode='stream')

    def test_time_encoded_array_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            create_time_encoded_array(self.video, mode='bogus')

    def test_time_encoded_frame_backends_equivalent(self):
        from wormtrails.processing import create_time_encoded_frame
        motion = subtract_average(self.video, use_absolute_difference=True, in_place=False)
        seq = create_time_encoded_frame(motion, window=3, start_time=2, mode='sequential')
        vec = create_time_encoded_frame(motion, window=3, start_time=2, mode='vectorized')
        self.assertTrue(np.array_equal(seq, vec))

    def test_time_encoded_frame_unknown_mode_raises(self):
        from wormtrails.processing import create_time_encoded_frame
        with self.assertRaises(ValueError):
            create_time_encoded_frame(self.video, mode='stream')

if __name__ == '__main__':
    unittest.main()
