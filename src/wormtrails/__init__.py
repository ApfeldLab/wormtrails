"""
wormtrails — analyse and visualise C. elegans motion on solid media.

Provides video I/O, vignetting correction, background subtraction,
time-encoded trail visualisation, automatic worm counting, chemotaxis
measurement, and an optional SQLite storage backend.
"""

from wormtrails.file_io import (
    read_video_file,
    write_mp4,
    write_avi
)

from wormtrails.display import (
    show_video_array,
    show_frame,
    show_time_encoding,
    count_assist,
    select_bait_spot
)

from wormtrails.processing import (
    correct_vignetting,
    subtract_average,
    fit_pixel_linear_model,
    create_track_array,
    create_time_encoded_array,
    create_time_encoded_array_parallel,
    create_time_encoded_frame,
    create_time_encoded_frame_vectorized,
    add_timestamp,
    align_frames
)

from wormtrails.quantitative import (
    Calibration,
    count_video,
    count_simple,
    create_plate_mask,
    measure_chemotaxis,
    measure_chemotaxis_parallel,
    calculate_relative_metrics,
    measure_component,
    measure_window
)

from wormtrails.streaming import (
    show_motion,
    show_time_encoded_frame,
    measure_chemotaxis_streaming,
    create_time_encoded_array_streaming,
    get_average_frame,
    get_motion,
    get_time_encoded_frame
)

from wormtrails.database import (
    create_database,
    write_measurements,
    read_measurements,
    add_recording,
    list_tables,
    SCHEMA
)

from wormtrails.colormaps import (
    black,
    white,
    white_to_black,
    black_to_white,
    blue_to_red,
    banded_blue_to_red,
    dark_separated_blue_to_red,
    middle_grey_last_black,
    hsv_rainbow
)

from wormtrails.memory import (
    available_memory_bytes,
    should_stream,
    estimated_time_encoding_bytes,
    estimated_binary_bytes
)

from wormtrails.gui import main as start_gui