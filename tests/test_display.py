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
        self.assertIn('phenotype', result.columns)

    def test_count_assist_dialog_marker_phenotype(self):
        from unittest import mock
        from wormtrails.display import _CountAssistDialog
        dialog = _CountAssistDialog.__new__(_CountAssistDialog)
        dialog.markers = []
        dialog.current_idx = 1
        dialog.num_frames = 3
        dialog.overlay_video = [None] * 3
        with mock.patch.object(dialog, '_render_frame'):
            dialog.current_phenotype = 'r'
            dialog._add_marker(10, 20, 1)
            dialog.current_phenotype = None
            dialog._add_marker(30, 40, 2)
        self.assertEqual(dialog.markers[0]['phenotype'], 'r')
        self.assertIsNone(dialog.markers[1]['phenotype'])


if __name__ == '__main__':
    unittest.main()
