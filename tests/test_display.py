import unittest
import numpy as np
from wormtrails.display import count_assist


class TestDisplay(unittest.TestCase):
    def test_count_assist_identical_frames_raises(self):
        video = np.ones((10, 100, 100), dtype=np.uint8) * 200
        with self.assertRaises(ValueError) as ctx:
            count_assist(video)
        self.assertIn("identical", str(ctx.exception))

    def test_count_assist_empty_video(self):
        video = np.zeros((0, 100, 100), dtype=np.uint8)
        import pandas as pd
        result = count_assist(video)
        self.assertTrue(result.empty)
        self.assertIn('worm_id', result.columns)


if __name__ == '__main__':
    unittest.main()
