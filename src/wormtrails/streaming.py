import cv2
import numpy as np
import pandas as pd

from wormtrails.quantitative import measure_window, calculate_relative_metrics

__all__ = [
    'show_motion',
    'show_time_encoded_frame',
    'measure_chemotaxis_streaming',
    'create_time_encoded_array_streaming',
    'get_average_frame',
    'get_motion',
    'get_time_encoded_frame',
]


def show_motion(
    video_path,
    worm_length=10,
    scale_factor=1,
    offset=0,
):
    """
    Streams a video from disk and displays a live motion preview for each frame.

    Unlike the in-memory pipeline (which loads the full video into a Numpy
    array), this helper reads frames one at a time from the source file, so
    it can be used for videos that are too large to hold in memory at once.

    Args:
        video_path: String path to the video file.
        worm_length: Integer typical length of a worm in pixels. Used as the
            kernel radius for the high-pass motion filter. Default is 10.
        scale_factor: Float value multiplied into each motion frame to boost
            contrast. Default is 1.
        offset: Float value added to each motion frame; negative values
            counteract noise. Default is 0.

    Returns:
        None. Opens an interactive OpenCV window and blocks until a key is
        pressed.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    average_frame = get_average_frame(cap)

    for motion_frame in get_motion(
        cap,
        reference_frame=average_frame,
        kernel_radius=worm_length,
    ):
        # Scale and shift the motion frame to improve visibility
        motion = motion_frame * scale_factor
        motion += offset

        # Clip and convert to 8-bit unsigned integer for display
        vis = np.clip(motion, 0, 255).astype(np.uint8)
        cv2.imshow("vis", vis)
        cv2.waitKey(1)

    # Release resources
    cv2.destroyAllWindows()
    cap.release()


def show_time_encoded_frame(
    video_path,
    worm_length,
    scale_factor=1,
    offset=0,
    start_frame=0,
    window=None,
    colormap=np.array([[0, 0, 0]]),
    light_background=True,
):
    """
    Streams a time-encoded (trail) frame from a video file and displays it.

    This is the streaming counterpart to the in-memory
    :func:`~wormtrails.create_time_encoded_frame`. It reads frames one at a
    time from disk and never materialises the full video array, making it
    suitable for very large files.

    Args:
        video_path: String path to the video file.
        worm_length: Integer typical length of a worm in pixels, used as the
            kernel radius for the high-pass motion filter.
        scale_factor: Float value multiplied into each motion frame to boost
            contrast. Default is 1.
        offset: Float value added to each motion frame; negative values
            counteract noise. Default is 0.
        start_frame: Integer index of the first frame of the trail window.
            Default is 0.
        window: Integer number of frames to include in the trail. If None,
            defaults to the total number of frames in the video. Default is
            None.
        colormap: Numpy array of shape (N, 3) of BGR color values applied to
            the trail, in the order they should trace (first to last frame).
            Default is black.
        light_background: Boolean. If True, assumes dark trails on a light
            background and inverts the rendering. Default is True.

    Returns:
        None. Opens an interactive OpenCV window and blocks until a key is
        pressed.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    average_frame = get_average_frame(cap)

    vis = get_time_encoded_frame(
        cap,
        reference_frame=average_frame,
        kernel_radius=worm_length,
        scale_factor=scale_factor,
        offset=offset,
        start_frame=start_frame,
        window=window,
        colormap=colormap,
        light_background=light_background,
    )

    cv2.imshow("vis", vis)
    cv2.waitKey(0)

    # Release resources
    cv2.destroyAllWindows()
    cap.release()


def measure_chemotaxis_streaming(
    video_path,
    thresh=3,
    worm_length=10,
    window=10,
    interval=60,
    minimum_size=10,
    maximum_size=1000,
    test_spot=None,
    calibration=None,
):
    """
    Measures chemotaxis metrics while streaming a video from disk.

    This is the streaming counterpart to
    :func:`~wormtrails.measure_chemotaxis`. Because data is read from the
    file one window at a time, the full binary array is never held in memory.
    Motion detection differs from the in-memory path: it thresholds the
    absolute difference between each frame and the average frame rather than
    consuming a pre-built binary array.

    Args:
        video_path: String path to the video file.
        thresh: Integer motion threshold; pixels whose absolute difference
            from the average frame exceeds this are treated as worm motion.
            Default is 3.
        worm_length: Integer typical length of a worm in pixels, used as the
            kernel radius for the high-pass motion filter. Default is 10.
        window: Integer number of frames per sliding analysis window.
            Default is 10.
        interval: Integer frame step between the starts of consecutive
            windows. Default is 60.
        minimum_size: Integer minimum 2D projection area (pixels) for a
            component to be counted as a worm. Default is 10.
        maximum_size: Integer maximum 2D projection area (pixels) for a
            component to be counted as a worm. Default is 1000.
        test_spot: Tuple or array of (y, x) absolute coordinates of the bait/
            test spot. If None (default), relative-angle metrics are omitted.
        calibration: Optional :class:`~wormtrails.Calibration` for converting
            pixel/frame units to physical units. Default is None.

    Returns:
        pandas.DataFrame with one row per detected worm trail per window.
        Includes position, direction and speed, plus r/theta/relative_angle
        when test_spot is given and physical units when calibration is given.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    average_frame = get_average_frame(cap)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    worm_data = []

    for t in range(0, n_frames - window + 1, interval):
        print(t)
        binary_array = []
        for motion_frame in get_motion(
            cap,
            reference_frame=average_frame,
            kernel_radius=worm_length,
            start_frame=t,
            window=window,
        ):
            # Threshold the motion frame into a binary worm mask
            binary = np.zeros_like(motion_frame, dtype=np.uint8)
            binary[motion_frame > thresh] = 255
            binary_array.append(binary)

        binary_array = np.stack(binary_array, axis=0)
        window_metrics = measure_window(
            binary_array, window, minimum_size, maximum_size, calibration=calibration,
        )

        for m in window_metrics:
            m['time'] = t
            # Calculate chemotaxis-specific metrics from position and direction
            pos = np.array([m['y'], m['x']])
            direction = np.array([m['direction_y'], m['direction_x']])

            if test_spot is not None:
                r, theta, rel_angle = calculate_relative_metrics(pos, direction, test_spot)
                m['r'] = r
                m['theta'] = theta
                m['relative_angle'] = rel_angle
                if calibration is not None:
                    m['r_mm'] = calibration.distance_mm(r)

            worm_data.append(m)

    return pd.DataFrame(worm_data)


def get_average_frame(cap):
    """
    Calculates the average (mean) frame of a video from a VideoCapture object.

    The average frame serves as a stationary reference for motion detection:
    each subsequent frame is compared against it so that only moving pixels
    are highlighted.

    Args:
        cap: OpenCV VideoCapture object for the video file to average.

    Returns:
        2D Numpy array (Y, X) of 64 bit floating point numbers (float64).

    Raises:
        ValueError: If no frames could be read from the video.
    """
    # Reset the frame pointer to the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sum_frame = None
    n_frames = 0
    while True:
        print(60 * " ", end="\r")
        print(n_frames, end="\r")
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sum_frame is None:
            sum_frame = np.zeros_like(frame, dtype=np.float64)
        sum_frame += frame.astype(np.float64)
        n_frames += 1

    # Guard against empty videos (checked after the loop, since n_frames is
    # only guaranteed >= 1 once at least one frame has been read)
    if n_frames == 0:
        raise ValueError("No frames read from video file")

    average_frame = sum_frame / n_frames

    return average_frame.astype(np.float64)


def get_motion(
    cap,
    reference_frame,
    kernel_radius=21,
    start_frame=0,
    window=None,
):
    """
    Yields a motion frame for each frame of a video, read lazily from disk.

    Each motion frame is the absolute difference between the brightness-
    normalised raw frame and the brightness-normalised stationary reference
    frame, after a median high-pass filter evens out large-scale illumination.

    Args:
        cap: OpenCV VideoCapture object for the video file.
        reference_frame: 2D Numpy array (Y, X), typically the average frame.
        kernel_radius: Integer; brightness variations larger than this will be
            evened out by the median filter. Default is 21.
        start_frame: Integer index of the first frame to process. Default is 0.
        window: Integer number of frames to yield. If None, yields until the
            end of the video. Default is None.

    Yields:
        2D Numpy array of 64 bit floating point numbers (float64) for each
        frame.

    Notes:
        The reference frame must have the same corrections applied to it as
        the video frames, which this function does internally via the median
        blur.
    """
    reference_frame = reference_frame.copy().astype(np.float64)

    # Blur the reference and normalise it the same way as live frames so it
    # becomes a true stationary background reference
    blur_frame = cv2.medianBlur(
        reference_frame.copy().astype(np.uint8), ksize=kernel_radius * 2 + 1
    ).astype(np.float64)
    target_brightness = np.mean(reference_frame).astype(np.float64)

    reference_frame *= target_brightness / blur_frame

    # Reset the frame pointer to the requested start frame
    frame_number = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if window is not None and frame_number >= start_frame + window:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Normalise brightness
        frame *= target_brightness / np.mean(frame)

        # Even out large scale brightness variation (effective high pass filter)
        frame *= target_brightness / blur_frame

        # Calculate absolute difference from the stationary reference frame
        motion = np.abs(frame - reference_frame)

        yield motion

        frame_number += 1


def get_time_encoded_frame(
    cap,
    reference_frame,
    kernel_radius=21,
    scale_factor=1,
    offset=0,
    start_frame=0,
    window=None,
    colormap=np.array([[0, 0, 0]]),
    light_background=True,
):
    """
    Builds a single time-encoded (trail) frame from a video, read from disk.

    This is the streaming counterpart to the in-memory
    :func:`~wormtrails.create_time_encoded_frame`: motion frames are consumed
    one at a time from the VideoCapture and projected together, so the whole
    video array is never held in memory.

    Args:
        cap: OpenCV VideoCapture object for the video file.
        reference_frame: 2D Numpy array (Y, X) used as the stationary reference.
        kernel_radius: Integer kernel radius for the high-pass motion filter.
            Default is 21.
        scale_factor: Float value multiplied into each motion frame to boost
            contrast. Default is 1.
        offset: Float value added to each motion frame; negative values
            counteract noise. Default is 0.
        start_frame: Integer index of the first frame of the trail window.
            Default is 0.
        window: Integer number of frames to project into the trail. If None,
            defaults to the total number of remaining frames. Default is None.
        colormap: Numpy array of shape (N, 3) of BGR color values applied to
            the trail in temporal order. Default is black.
        light_background: Boolean. If True, assumes dark trails on a light
            background and inverts the rendering. Default is True.

    Returns:
        3D Numpy array of 8 bit unsigned integers (uint8) with shape
        (height, width, 3) containing the time-encoded trail frame.
    """
    if window is None:
        window = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame

    time_encoded_frame = None
    t = 0

    for motion_frame in get_motion(
        cap,
        reference_frame=reference_frame,
        kernel_radius=kernel_radius,
        start_frame=start_frame,
        window=window,
    ):
        motion = motion_frame

        # Multiply motion by scale_factor and add offset to improve visibility
        motion *= scale_factor
        motion += offset

        # Get the colour for the current frame from the colormap
        tmap = colormap[int(np.shape(colormap)[0] * t / window), :]
        if light_background:  # Invert colour if final background is light
            tmap = 255 - tmap

        # Colour each pixel by the colormap and scale brightness by motion
        colormapped_frame = np.clip(
            cv2.merge([tmap[0] * motion / 255, tmap[1] * motion / 255, tmap[2] * motion / 255]),
            0, 255,
        ).astype(np.uint8)
        if light_background:  # Invert the whole frame for a light background
            colormapped_frame = 255 - colormapped_frame

        # Initialise or accumulate the time-encoded frame
        if time_encoded_frame is None:
            time_encoded_frame = colormapped_frame
        elif light_background:
            # Minimum projection keeps dark tracks on a light background
            time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0)
        else:
            # Maximum projection keeps light tracks on a dark background
            time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0)

        t += 1

    return time_encoded_frame


def create_time_encoded_array_streaming(
    video_path,
    save_path=None,
    worm_length=10,
    scale_factor=1,
    offset=0,
    window=1,
    colormap=np.array([[0, 0, 0]]),
    light_background=True,
):
    """
    Creates a time-encoded video directly from a file, streaming from disk.

    Rather than loading the entire scan into memory (as the in-memory
    :func:`~wormtrails.create_time_encoded_array` does), this function reads
    frames incrementally, computes each trail frame, and either previews it
    interactively and/or writes it to an MP4 file. Suitable for scans too
    large to fit in memory.

    Args:
        video_path: String path to the source video file.
        save_path: Optional string path to write the resulting time-encoded
            video to as MP4. If None, the output is only previewed in an
            OpenCV window. Default is None.
        worm_length: Integer typical length of a worm in pixels, used as the
            kernel radius for the high-pass motion filter. Default is 10.
        scale_factor: Float value multiplied into each motion frame to boost
            contrast. Default is 1.
        offset: Float value added to each motion frame; negative values
            counteract noise. Default is 0.
        window: Integer number of frames per trail. Default is 1.
        colormap: Numpy array of shape (N, 3) of BGR color values applied to
            each trail in temporal order. Default is black.
        light_background: Boolean. If True, assumes dark trails on a light
            background and inverts the rendering. Default is True.

    Returns:
        None. The output is previewed interactively and/or written to
        save_path.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        fps = 60
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    average_frame = get_average_frame(cap)
    blur_frame = cv2.medianBlur(
        average_frame.copy().astype(np.uint8), ksize=worm_length * 2 + 1
    ).astype(np.float64)
    target_brightness = np.mean(average_frame).astype(np.float64)

    # Normalise the average frame the same way as live frames so it acts as a
    # true stationary background reference
    average_frame *= target_brightness / blur_frame

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for t_global in range(n_frames - window):
        # Reset the frame pointer to the start of the trail window
        frame_number = t_global
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        time_encoded_frame = None
        # Loop through and process each frame in the trail window
        for t in range(window):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Normalise brightness
            frame *= target_brightness / np.mean(frame)

            # Even out large scale brightness variation (effective high pass filter)
            frame *= target_brightness / blur_frame

            # Calculate absolute difference from the stationary reference frame
            motion = np.abs(frame - average_frame)

            # Multiply motion by scale_factor and add offset to improve visibility
            motion *= scale_factor
            motion += offset

            # Get the colour for the current frame from the colormap
            tmap = colormap[int(np.shape(colormap)[0] * t / window), :]
            if light_background:  # Invert colour if final background is light
                tmap = 255 - tmap

            # Colour each pixel by the colormap and scale brightness by motion
            colormapped_frame = np.clip(
                cv2.merge([tmap[0] * motion / 255, tmap[1] * motion / 255, tmap[2] * motion / 255]),
                0, 255,
            ).astype(np.uint8)
            if light_background:  # Invert the whole frame for a light background
                colormapped_frame = 255 - colormapped_frame

            # Initialise or accumulate the time-encoded frame
            if time_encoded_frame is None:
                time_encoded_frame = colormapped_frame
            elif light_background:
                # Minimum projection keeps dark tracks on a light background
                time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0)
            else:
                # Maximum projection keeps light tracks on a dark background
                time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0)

        # Preview the frame and optionally write it to the output video
        if time_encoded_frame is not None:
            cv2.imshow("vis", time_encoded_frame)
            if save_path is not None:
                out.write(time_encoded_frame)
            cv2.waitKey(1)

    # Release resources
    cv2.destroyAllWindows()
    if save_path is not None:
        out.release()
    cap.release()
