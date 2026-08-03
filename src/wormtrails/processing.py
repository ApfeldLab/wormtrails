import cv2
import numpy as np
from joblib import Parallel, delayed

__all__ = [
    'correct_vignetting',
    'subtract_average',
    'fit_pixel_linear_model',
    'create_track_array',
    'create_time_encoded_array',
    'create_time_encoded_array_parallel',
    'create_time_encoded_frame',
    'create_time_encoded_frame_vectorized',
    'add_timestamp',
    'align_frames',
]

def correct_vignetting(array, kernel_size=None, use_median_blur=True, in_place=False):
    """
    Corrects vignetting in a video array by normalizing each frame to have the same brightness as the average frame and dividing by a blur of the average frame.
    If a single frame (2D array) is provided, the frame brightness adjustment step is skipped since no normalization is needed.
    This step can be skipped if initial scan is evenly lit and the lens used does not vignette.
    Kernel size must be odd; smaller values make smaller brightness variations get evened out.

    Args:
        array: 2D or 3D Numpy array containing the video frames. If 3D, time is axis 0. For 2D arrays, only vignetting correction (no brightness adjustment) is applied.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame. If None (default), the kernel size is calculated as one quarter the smaller image dimension (min of width or height), rounded up and made odd.
        use_median_blur: Boolean value. If True (default), uses a median filter to create the blurred frame for background brightness estimation. If False, uses Gaussian blur.
        in_place: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 2D or 3D Numpy array of 8 bit unsigned integers (uint8) containing the corrected video frames. Returns None if in_place=True.
    """
    if not in_place:
        array = array.copy()

    # Handle 2D and 3D arrays by wrapping 2D in a new axis for unified processing
    is_single_frame = array.ndim == 2
    if is_single_frame:
        average_frame = array.copy()
    else:
        average_frame = np.mean(array, axis=0)
    
    if kernel_size is None:
        kernel_size = int(min(average_frame.shape[:2])/8) * 2 + 1 # Kernel size is one quarter the smaller image dimension
    if use_median_blur:
        blur_frame = cv2.medianBlur(average_frame.astype(np.uint8), kernel_size)
    else:
        blur_frame = cv2.GaussianBlur(average_frame, (kernel_size,kernel_size), 0)
    
    target_brightness = np.mean(average_frame)

    if is_single_frame:
        array = (array * target_brightness / blur_frame).astype(np.uint8)
    else:
        # loop through and correct each frame
        for i in range(array.shape[0]):
            frame = array[i].astype(np.float32)
            frame_brightness = np.mean(frame)

            if frame_brightness != 0:
                scaled_frame = frame * (target_brightness / frame_brightness) # scale each frame to have the same brightness as the average frame
            else:
                scaled_frame = frame  # Avoid division by zero

            # divide each frame by the blur of the average frame to correct vignetting or other large scale brightness variations
            array[i] = (scaled_frame * target_brightness / blur_frame).astype(np.uint8)

    if not in_place:
        return array

def subtract_average(video_array, average_start=0, average_end=-1, use_absolute_difference=True, use_projection=False, light_background=True, in_place=False):
    """
    Subtracts the average frame or min/max projected frame from each frame in the video array.
    This effectively only shows pixels which change in value over the course of the recording.
    A range of frames can be specified to calculate the average frame, and an absolute difference can be used instead of clipped subtraction.
    An ideal averaging range contains no stationary worms, as these will appear in negative space after subtraction.

    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        average_start: Integer value for the start index of the frame range used to calculate the average frame. Default is 0.
        average_end: Integer value for the end index of the frame range used to calculate the average frame. If -1 (default), uses the last frame.
        use_absolute_difference: Boolean value. If True, uses absolute difference between scaled frames and average, making motion symmetric. If False, uses one-sided subtraction based on light_background.
        use_projection: Boolean value. If True, uses a min/max projection instead of mean for the average frame, which is useful when stationary background elements need to be preserved.
        light_background: Boolean value. If True, assumes dark worms on light background and uses inverted subtraction. If False, assumes bright objects on dark background.
        in_place: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the subtracted video frames with only motion visible, with time as axis 0. Returns None if in_place=True.
    """
    if not in_place:
        video_array = video_array.copy()

    if average_end == -1:
        average_end = video_array.shape[0]
    
    if not use_projection:
        average_frame = np.mean(video_array[average_start:average_end,:,:], axis=0)
    elif light_background:
        average_frame = np.max(video_array[average_start:average_end,:,:], axis=0)
    else:
        average_frame = np.min(video_array[average_start:average_end,:,:], axis=0)
    
    target_brightness = np.mean(average_frame)

    # loop through each frame to correct and subtract the average frame, leaving only motion visible
    for i in range(video_array.shape[0]):
        frame = video_array[i].astype(np.float32)
        frame_brightness = np.mean(frame) # brightness correction is performed in case vignetting correction was not used

        if frame_brightness != 0:
            scaled_frame = frame * (target_brightness / frame_brightness) # scale each frame to have the same brightness as the average frame
        else:
            scaled_frame = frame # Avoid division by zero

        # perform subtraction
        if use_absolute_difference:
            video_array[i] = np.abs(scaled_frame - average_frame).astype(np.uint8)
        elif light_background:
            video_array[i] = np.clip(average_frame - scaled_frame, 0, None).astype(np.uint8)
        else:
            video_array[i] = np.clip(scaled_frame - average_frame, 0, None).astype(np.uint8)

    if not in_place:
        return video_array

def fit_pixel_linear_model(video: np.ndarray):
    """
    Fit a linear model to each pixel's intensity over time.

    Returns residual array, slope, and intercept for each pixel.

    Args:
        video: 3D Numpy array of shape (T, H, W) containing the video frames,
            with time as axis 0.

    Returns:
        residuals: 3D Numpy array of shape (T, H, W) containing the residuals
            for each pixel (float16).
        slope: 2D Numpy array of shape (H, W) (float64).
        intercept: 2D Numpy array of shape (H, W) (float64).

    Raises:
        ValueError: If the input array is not 3D.
        ValueError: If fewer than 2 frames are provided.
    """
    if video.ndim != 3:
        raise ValueError("Input video must be a 3D array (T, H, W)")
    T, H, W = video.shape
    if T < 2:
        raise ValueError("Need at least 2 frames to fit a linear model")

    y_mat = video.reshape(T, -1)          # (T, H*W)
    t = np.arange(T, dtype=np.float64).reshape(-1, 1)
    X = np.column_stack((t, np.ones(T, dtype=np.float64)))  # (T, 2)

    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ X.T @ y_mat          # (2, H*W)

    slope = beta[0].reshape(H, W)
    intercept = beta[1].reshape(H, W)

    # residuals in float16 to reduce memory (half the footprint of float32)
    y_pred = X.astype(np.float16) @ beta.astype(np.float16)  # (T, H*W) float16
    residuals = (y_mat.astype(np.float16) - y_pred).reshape(T, H, W)

    return residuals, slope, intercept

def create_track_array(average_subtracted_array, window):
    """
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create frames with trails.
    This is the simplest implementation of trails, and does not support colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the average subtracted video frames, with time as axis 0.
        window: Integer value for the window size (number of frames to look back), used to create trails by projecting motion within this window.

    Returns:
        track_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the track array with trails, with time as axis 0.

    Notes:
        Uses maximum projection to create trails, assuming a dark background. This is the simplest trail implementation.
    """
    track_array = []

    # loop through each frame and project from the windows starting at that frame
    for i in range(average_subtracted_array.shape[0]):
        track_frame = np.max(average_subtracted_array[i:i+window,:,:], axis=0) # project with maximum function since a dark background is used
        track_array.append(track_frame)

    return np.stack(track_array, axis=0)

def _create_time_encoded_array_sequential(average_subtracted_array, colormap, window, scale_factor, offset, light_background):
    """
    Sequential in-memory implementation of time-encoded trail creation.

    Loops over every frame position and builds one trail frame per window
    start. This is the reference implementation used by the dispatcher and is
    invoked directly by the parallel backend for small inputs.

    Args:
        average_subtracted_array: 3D Numpy array of uint8, time as axis 0.
        colormap: Numpy array of shape (N, 3) of BGR colors.
        window: Integer number of frames per trail.
        scale_factor: Float brightness scaling factor.
        offset: Float brightness offset.
        light_background: Boolean; if True, renders dark trails on light.

    Returns:
        4D Numpy array of uint8 with shape (n_frames, H, W, 3).
    """
    time_encoded_array = []

    # Loop through the average subtracted array and create time encoded frames
    # for the windows starting at each frame
    for i in range(average_subtracted_array.shape[0] - window):
        frame = create_time_encoded_frame(
            average_subtracted_array,
            colormap=colormap,
            window=window,
            start_time=i,
            scale_factor=scale_factor,
            offset=offset,
            light_background=light_background,
        )
        time_encoded_array.append(frame)

    return np.stack(time_encoded_array, axis=0)


def create_time_encoded_array(source, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True, mode='auto', n_jobs=-1, save_path=None, worm_length=10):
    """
    Creates a time-encoded trail video from either an in-memory array or a file on disk.

    This is a dispatcher: depending on the input and the ``mode`` argument, it
    routes to one of several backend implementations.

    * ``str`` input (a video path) is routed to the streaming pipeline, which
      reads frames one at a time from disk and never materialises the full
      array. This is chosen automatically when the input is a path.
    * Numpy array input is processed in memory, routed between the
      ``sequential``, ``vectorized``, or ``parallel`` backends.

    Args:
        source: Either a 3D Numpy array (uint8, time as axis 0) of average
            subtracted frames, or a string path to a video file.
        colormap: Numpy array of shape (N, 3) of BGR colors applied to each
            trail frame to encode temporal information.
        window: Integer number of frames to look back for each trail. Shorter
            windows take less processing time; longer windows reveal motion
            patterns but suit less densely populated plates. Default is 20.
        scale_factor: Float scaling factor applied to trail brightness. Higher
            values increase contrast. Default is 1.
        offset: Float brightness offset; positive brightens, negative
            counteracts noise. Default is 0.
        light_background: Boolean. If True, renders dark trails on a light
            background; if False, bright trails on dark. Default is True.
        mode: String selecting the backend. One of 'auto' (default), 'sequential',
            'vectorized', 'parallel', or 'stream'. 'auto' streams for path input
            and otherwise chooses a parallel or sequential in-memory backend
            based on frame count. An explicit mode forces that backend.
        n_jobs: Integer number of parallel workers for the 'parallel' backend
            (-1 for all cores). Default is -1.
        save_path: Optional string path used only for streaming (path) input,
            to write the resulting time-encoded video to an MP4 file. If None,
            the streaming pipeline only previews interactively. Default is None.
        worm_length: Integer typical worm length in pixels, used only for
            streaming (path) input as the high-pass kernel radius. Default is 10.

    Returns:
        4D Numpy array of uint8 with shape (n_frames, H, W, 3) for in-memory
        input, or the return value of the streaming pipeline (which writes to
        ``save_path`` and previews) for path input.

    Raises:
        ValueError: If the explicit ``mode`` is incompatible with the input
            type (e.g. an in-memory backend given a path, or 'stream' given
            an array).
    """
    # A string source is a file path -> route to the streaming backend, which
    # avoids loading the whole video into memory at once.
    if isinstance(source, str):
        if mode not in ('auto', 'stream'):
            raise ValueError(
                f"mode={mode!r} requires an in-memory Numpy array, but a video "
                "path was given. Use mode='stream' or 'auto' for path input."
            )
        # Imported lazily to avoid a circular import (streaming imports
        # processing indirectly through quantitative).
        from wormtrails.streaming import create_time_encoded_array_streaming
        return create_time_encoded_array_streaming(
            source,
            save_path=save_path,
            worm_length=worm_length,
            scale_factor=scale_factor,
            offset=offset,
            window=window,
            colormap=colormap,
            light_background=light_background,
        )

    if mode == 'stream':
        raise ValueError(
            "mode='stream' requires a video path (str) as input, but a Numpy "
            "array was given. For in-memory input use 'sequential', "
            "'vectorized', 'parallel', or 'auto'."
        )

    # In-memory input: route between sequential/vectorized/parallel backends.
    valid = {'auto', 'sequential', 'vectorized', 'parallel'}
    if mode not in valid:
        raise ValueError(
            f"Unknown mode={mode!r}. Expected one of {sorted(valid)} or 'stream'."
        )

    if mode == 'vectorized':
        return np.stack(
            [
                create_time_encoded_frame_vectorized(
                    source, colormap, window, i, scale_factor, offset, light_background
                )
                for i in range(source.shape[0] - window)
            ],
            axis=0,
        )

    if mode == 'parallel' or (mode == 'auto' and source.shape[0] - window > 20):
        return create_time_encoded_array_parallel(
            source, colormap, window, scale_factor, offset, light_background, n_jobs
        )

    return _create_time_encoded_array_sequential(
        source, colormap, window, scale_factor, offset, light_background
    )

def create_time_encoded_array_parallel(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True, n_jobs=-1):
    """Parallel version of create_time_encoded_array using joblib.
    
    Each time-encoded frame is computed independently, making this ideal for parallelization.
    Falls back to the sequential version when the number of output frames is small
    (below the threshold) to avoid parallelization overhead.
    
    Args:
        average_subtracted_array: 3D Numpy array of shape (T, H, W) of uint8.
        colormap: Numpy array of shape (N, 3) with BGR color values.
        window: Window size for trail creation (default: 20).
        scale_factor: Brightness scaling factor (default: 1).
        offset: Integer or float brightness offset (default: 0).
        light_background: Light background flag (default: True).
        n_jobs: Number of parallel workers (-1 for all cores, default: -1).
        
    Returns:
        time_encoded_array: 4D Numpy array of shape (n_frames, H, W, 3) of uint8.
    """
    n_frames = average_subtracted_array.shape[0] - window
    
    # Use sequential mode for small arrays to avoid parallelization overhead
    if n_frames <= 20:
        return _create_time_encoded_array_sequential(
            average_subtracted_array, colormap, window, scale_factor, offset, light_background
        )
    
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(create_time_encoded_frame)(average_subtracted_array, colormap, window, i, scale_factor, offset, light_background)
        for i in range(n_frames)
    )
    
    return np.stack(results, axis=0)

def create_time_encoded_frame(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, start_time=0, scale_factor=1, offset=0, light_background=True, mode='auto'):
    """
    Creates a single time-encoded (trail) frame from average subtracted frames.

    This is a dispatcher between the sequential in-memory implementation
    (default) and the vectorized implementation, selected via ``mode``:
    'auto' and 'sequential' use the slower, memory-lean loop; 'vectorized'
    uses the faster but memory-heavier implementation.

    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers
            (uint8) containing the average subtracted video frames, with time
            as axis 0.
        colormap: Numpy array of shape (N, 3) containing the colormap colors
            (B, G, R values 0-255), applied to the trail frame with color
            values applied in order scaled to the window size.
        window: Integer number of frames to look back, used to create trails.
        start_time: Integer start time of the frame. The window starts at this
            frame and continues forward for the specified window size.
            Default is 0.
        scale_factor: Float scaling factor applied to trail brightness. Higher
            values increase contrast. Default is 1.
        offset: Float brightness offset; positive brightens, negative
            counteracts noise. Default is 0.
        light_background: Boolean. If True, renders dark trails on a light
            background; if False, bright trails on dark. Default is True.
        mode: String selecting the backend. One of 'auto' (default),
            'sequential', or 'vectorized'.

    Returns:
        time_encoded_frame: 3D Numpy array of uint8 with shape (height, width,
            3) containing the time encoded frame with trails.

    Raises:
        ValueError: If ``mode`` is not a valid backend name.
    """
    if mode in ('auto', 'sequential'):
        return _create_time_encoded_frame_sequential(
            average_subtracted_array, colormap, window, start_time,
            scale_factor, offset, light_background,
        )
    if mode == 'vectorized':
        return create_time_encoded_frame_vectorized(
            average_subtracted_array, colormap, window, start_time,
            scale_factor, offset, light_background,
        )
    raise ValueError(
        f"Unknown mode={mode!r}. Expected one of 'auto', 'sequential', "
        "'vectorized'."
    )


def _create_time_encoded_frame_sequential(average_subtracted_array, colormap, window, start_time, scale_factor, offset, light_background):
    """
    Sequential in-memory implementation of a single time-encoded frame.

    Loops over the window, building a colormapped frame per input frame and
    combining them with a minimum (light background) or maximum (dark
    background) projection.

    Args:
        average_subtracted_array: 3D Numpy array of uint8, time as axis 0.
        colormap: Numpy array of shape (N, 3) of BGR colors.
        window: Integer number of frames per trail.
        start_time: Integer index of the first frame of the window.
        scale_factor: Float brightness scaling factor.
        offset: Float brightness offset.
        light_background: Boolean; True renders dark trails on light.

    Returns:
        3D Numpy array of uint8 with shape (height, width, 3).
    """
    time_encoded_frame = None

    # Loop through the window and create colormapped frames
    for t in range(window):
        tmap = colormap[int(np.shape(colormap)[0] * t / window), :]  # get the color for the current frame from the colormap
        if light_background:  # invert the color used for colormapping if using a light background since it's going to be inverted back later
            tmap = 255 - tmap

        # Set the color of each pixel based on the colormap and the brightness based on the corresponding average subtracted frame
        colormapped_frame = np.clip(
            cv2.merge([
                tmap[0] * average_subtracted_array[start_time + t] / 255,
                tmap[1] * average_subtracted_array[start_time + t] / 255,
                tmap[2] * average_subtracted_array[start_time + t] / 255,
            ]) * scale_factor + offset, 0, 255,
        ).astype(np.uint8)
        if light_background:  # invert the colormapped frame if a light background is desired
            colormapped_frame = 255 - colormapped_frame

        # Initialize or update the time encoded frame
        if time_encoded_frame is None:
            time_encoded_frame = colormapped_frame
        elif light_background:
            # Minimum pixel value is used for projections if we want dark tracks on a light background
            time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0)
        else:
            # Maximum pixel value is used for projections if we want light tracks on a dark background
            time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0)

    return time_encoded_frame

def create_time_encoded_frame_vectorized(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, start_time=0, scale_factor=1, offset=0, light_background=True):
    """
    Faster than create_time_encoded_frame but uses more memory.
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create a single frame with trails.
    This implementation supports colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array of shape (N, 3) containing the colormap colors (B, G, R values 0-255), applied to the trail frame with color values being applied in order scaled to the window size.
        window: Integer value for the window size (number of frames to look back), used to create trails. Shorter windows take less processing time.
        start_time: Integer value for the start time of the frame. The window starts at this frame and continues forward for the specified window size.
        scale_factor: Float value for the scaling factor, applied to the trails to adjust brightness. Higher values increase contrast. Default is 1.
        offset: Integer or float value for the brightness offset, positive values brighten the image, negative values darken it and can counteract noise. Default is 0.
        light_background: Boolean value. If True, assumes dark trails on light background and inverts the trail rendering. If False, assumes bright trails on dark background.

    Returns:
        time_encoded_frame: 3D Numpy array of 8 bit unsigned integers (uint8) containing the time encoded frame with trails, with color channel as axis 3 (last axis), shape (height, width, 3).

    Notes:
        Uses numpy minimum/maximum projection to combine trail information from multiple frames.

        When using symmetric (grayscale) colormaps such as white_to_black or black_to_white,
        flipping light_background is equivalent to using the opposite colormap.
        For asymmetric colormaps (e.g. blue_to_red), light_background flips the color channel
        values, producing visually distinct (complementary) output rather than identical results.
        To reverse temporal ordering regardless of colormap, swap it for its inverse
        (e.g. blue_to_red for red_to_blue, white_to_black for black_to_white).
    """
    time_encoded_frame = None

    # vectorized colormap lookup and per-frame color scaling
    colormap_indices = np.arange(window)
    colormap_indices = (np.array(colormap_indices) * np.shape(colormap)[0] / window).astype(int)
    tmap = colormap[colormap_indices, :]  # shape (window, 3)

    if light_background:
        tmap = 255 - tmap

    frames = average_subtracted_array[start_time:start_time + window]  # (window, H, W)
    colormapped_frames = np.clip(tmap[:, np.newaxis, np.newaxis, :] * frames[:, :, :, np.newaxis] / 255 * scale_factor + offset, 0, 255).astype(np.uint8)  # (window, H, W, 3)

    if light_background:
        colormapped_frames = 255 - colormapped_frames

    # sequential accumulation (can't vectorize min/max projection)
    for i in range(window):
        if time_encoded_frame is None:
            time_encoded_frame = colormapped_frames[i]
        elif light_background:
            time_encoded_frame = np.minimum(time_encoded_frame, colormapped_frames[i])
        else:
            time_encoded_frame = np.maximum(time_encoded_frame, colormapped_frames[i])

    return time_encoded_frame

def add_timestamp(video_array, black_background=True, font_scale=2, font_thickness=2, seconds_per_frame=1, in_place=False):
    """
    Adds a timestamp to each frame of the video array.
    This should be done last in the processing pipeline, after time encoding returns a color array.
    
    Args:
        video_array: 4D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with shape (time, height, width, 3), where time is axis 0 and color channels are the last axis.
        black_background: Boolean value. If True, uses white text on the assumption of a black background. If False, uses black text for a light background.
        font_scale: Float value for the OpenCV font scale. Controls the size of the timestamp text. Default is 2.
        font_thickness: Integer value for the OpenCV font thickness. Default is 2.
        seconds_per_frame: Float value for the frame time in seconds. This is the actual captured frame time (not the desired export frame time). Default is 1 second per frame.
        in_place: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 4D Numpy array of 8 bit unsigned integers (uint8) containing the video frames with timestamps in bottom-left corner, with time as axis 0 and color channel as axis 3 (last axis). Returns None if in_place=True.

    Notes:
        Only accepts color (3-channel) input since cv2.putText requires a color image.
        Timestamps are formatted as MM:SS and placed in the bottom-left corner of each frame.
        Input array must have 4 dimensions (time, height, width, channels) with 3 channels.
    """
    if not in_place:
        video_array = video_array.copy()

    num_frames, height, width, channel = video_array.shape # only accepts color input, since the cv2.putText function requires a color image
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, height-10) # Bottom left corner
    if black_background:
        color = 255 # 255 for white text, 0 for black text
    else:
        color = 0

    # loop through and add timestamp to each frame
    for i in range(num_frames):
        total_seconds = int(i*seconds_per_frame)
        timestamp = f"{(total_seconds // 60):02d}:{(total_seconds % 60):02d}" # Timestamp format mm:ss
        cv2.putText(video_array[i], timestamp, position, font, font_scale, (color, color, color), font_thickness)

    if not in_place:
        return video_array

def align_frames(ref, target):
    """
    Aligns the target frame to the reference frame using ORB feature detection and homography estimation.
    
    Args:
        ref: 2D Numpy array of 8 bit unsigned integers (uint8) containing the reference frame.
        target: 2D Numpy array of 8 bit unsigned integers (uint8) containing the target frame to align.

    Returns:
        aligned: 2D Numpy array of 8 bit unsigned integers (uint8) containing the target frame warped to match the reference frame geometry.

    Raises:
        ValueError: If features cannot be detected in either frame, fewer than 15 good matches are found, or homography cannot be computed.

    Notes:
        Uses ORB feature detection with BFMatcher k-NN and Lowe's ratio test.
        Requires at least 15 good matches or raises ValueError.
        Prints diagnostic info (match count, inliers, homography matrix, rotation, translation, scale).
    """
    orb = cv2.ORB_create(nfeatures=5000)

    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(target, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not detect features")

    # Use BFMatcher with k-NN and ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)

    print(f"Filtered matches: {len(good)}")

    if len(good) < 15:
        raise ValueError("Not enough good matches")

    # Get matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Use homography with RANSAC to handle rotation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=5000)

    if H is None:
        raise ValueError("Could not find homography")

    inliers = mask.ravel() > 0
    print(f"Inliers: {np.sum(inliers)}/{len(inliers)}")

    # Convert homography to affine parameters
    # H is 3x3: [[a b c], [d e f], [0 0 1]]
    a, b, c = H[0, 0], H[0, 1], H[0, 2]
    d, e, f = H[1, 0], H[1, 1], H[1, 2]

    # Calculate rotation angle (use atan2(-b, a) for correct sign)
    angle = np.degrees(np.arctan2(-b, a))
    scale = np.sqrt(a**2 + b**2)

    print(f"Homography:\n{H}")
    print(f"Rotation: {angle:.2f}°, Translation: ({c:.1f}, {f:.1f}), Scale: {scale:.4f}")

    # Apply transformation
    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(
        target, 
        H, 
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned