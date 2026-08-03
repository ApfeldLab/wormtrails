"""
Memory-awareness helpers used to decide when to use a disk-streaming backend.

The in-memory backends (sequential / vectorized / parallel) load and process
the full video as a Numpy array. When available system memory is tight, it is
preferable to fall back to a streaming backend that reads frames one at a time
from disk. The decision rule used throughout the library is:

    stream if  available_memory < expected_bytes * 1.5

i.e. stream whenever there is less than 50% more memory available than the
in-memory backends are expected to consume.
"""

import numpy as np

__all__ = [
    'available_memory_bytes',
    'should_stream',
    'estimated_time_encoding_bytes',
    'estimated_binary_bytes',
]


def available_memory_bytes():
    """
    Best-effort estimate of currently available system memory in bytes.

    Uses ``psutil`` when installed. Returns ``None`` if a value cannot be
    determined, in which case callers should conservatively assume enough
    memory is available (so they do not stream unnecessarily).

    Returns:
        int or None: Number of free bytes of system memory, or None if unknown.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


def should_stream(expected_bytes):
    """
    Decide whether to use the disk-streaming backend for a given memory budget.

    Streaming is selected when the available memory is less than 50% more than
    the number of bytes the in-memory backends are expected to use. If the
    available memory cannot be determined, ``False`` is returned (assume enough
    memory is available).

    Args:
        expected_bytes: int number of bytes the in-memory backends are expected
            to consume.

    Returns:
        bool: True if the streaming backend should be used, False otherwise.
    """
    avail = available_memory_bytes()
    if avail is None:
        return False
    return avail < expected_bytes * 1.5


def estimated_time_encoding_bytes(source_shape, window):
    """
    Estimate the in-memory bytes the non-streaming time-encoding backends need.

    Accounts for the uint8 input video array, an intermediate working copy, and
    the uint8 (T, H, W, 3) time-encoded output array, with a margin factor.

    Args:
        source_shape: tuple of the input array shape (T, H, W).
        window: int number of frames per trail.

    Returns:
        int estimated number of bytes.
    """
    T, H, W = source_shape[:3]
    input_bytes = T * H * W                 # uint8 input
    out_frames = max(T - window, 0)
    output_bytes = out_frames * H * W * 3   # uint8 (T, H, W, 3) output
    return input_bytes * 2 + output_bytes * 2


def estimated_binary_bytes(shape):
    """
    Estimate the in-memory bytes the non-streaming chemotaxis backend needs.

    The binary array plus a working margin.

    Args:
        shape: tuple of the binary array shape (T, H, W).

    Returns:
        int estimated number of bytes.
    """
    return int(np.prod(shape)) * 2
