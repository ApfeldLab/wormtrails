import unittest
import types
import unittest.mock as mock
from wormtrails import memory


def _patch_psutil(available):
    ns = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=available)
    )
    return mock.patch.dict('sys.modules', {'psutil': ns})


class TestMemoryHelpers(unittest.TestCase):
    def test_estimated_time_encoding_bytes(self):
        # (T,H,W) = (10,100,100), window 5
        # input 10*100*100=100000, output 5*100*100*3=150000
        # 2*input + 2*output = 200000 + 300000 = 500000
        self.assertEqual(memory.estimated_time_encoding_bytes((10, 100, 100), 5), 500000)

    def test_estimated_binary_bytes(self):
        self.assertEqual(memory.estimated_binary_bytes((10, 100, 100)), 200000)

    def test_available_memory_none_without_psutil(self):
        with mock.patch.dict('sys.modules', {'psutil': None}):
            self.assertIsNone(memory.available_memory_bytes())

    def test_should_stream_none_available(self):
        self.assertFalse(memory.should_stream(10**6))

    def test_should_stream_when_memory_tight(self):
        # avail < expected * 1.5  -> stream
        with _patch_psutil(available=100_000_000):
            self.assertTrue(memory.should_stream(67_000_000))   # 67MB*1.5 = 100.5MB
            self.assertFalse(memory.should_stream(66_000_000))  # 66MB*1.5 = 99MB <= 100MB
            self.assertFalse(memory.should_stream(50_000_000))

    def test_should_stream_available_matches_threshold(self):
        # exactly 1.5x available -> not strictly less, so False
        with _patch_psutil(available=100_000_000):
            self.assertFalse(memory.should_stream(100_000_000 * 2 / 3))


if __name__ == '__main__':
    unittest.main()
