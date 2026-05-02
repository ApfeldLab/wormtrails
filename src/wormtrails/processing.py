import cv2
import numpy as np
import math

def correct_vignetting(array, kernel_size=None, use_median_blur=True, inPlace=False):
    """
    Corrects vignetting in a video array by normalizing each frame to have the same brightness as the average frame and dividing by a blur of the average frame.
    If a single frame (2D array) is provided, the frame brightness adjustment step is skipped since no normalization is needed.
    This step can be skipped if initial scan is evenly lit and the lens used does not vignette.
    Kernel size must be odd; smaller values make smaller brightness variations get evened out.

    Args:
        array: 2D or 3D Numpy array containing the video frames. If 3D, time is axis 0. For 2D arrays, only vignetting correction (no brightness adjustment) is applied.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame. If None (default), the kernel size is calculated as one quarter the image width, rounded up and made odd.
        use_median_blur: Boolean value. If True (default), uses a median filter to create the blurred frame for background brightness estimation. If False, uses Gaussian blur.
        inPlace: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 2D or 3D Numpy array of 8 bit unsigned integers (uint8) containing the corrected video frames. Returns None if inPlace=True.
    """
    if not inPlace:
        array = array.copy()

    # Handle 2D and 3D arrays by wrapping 2D in a new axis for unified processing
    is_single_frame = array.ndim == 2
    if is_single_frame:
        average_frame = array.copy()
    else:
        average_frame = np.mean(array, axis=0)
    
    if kernel_size is None:
        kernel_size = int(average_frame.shape[0]/8) * 2 + 1 # Kernel size is one quarter the image width
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

    if not inPlace:
        return array

def subtract_average(video_array, average_start=0, average_end=-1, use_absolute_difference=True, use_projection=False, light_background=True, inPlace=False):
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
        inPlace: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the subtracted video frames with only motion visible, with time as axis 0. Returns None if inPlace=True.
    """
    if not inPlace:
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

    if not inPlace:
        return video_array

def fit_pixel_linear_model(video: np.ndarray):
    """
    Fit a linear model to each pixel's intensity over time.

    Returns residual sum of squares (RSS), slope, and intercept for each pixel.

    Parameters
    ----------
    video : np.ndarray, shape (T, H, W)

    Returns
    -------
    residuals : np.ndarray, shape (T, H, W)
        Residuals for each pixel.
    slope : np.ndarray, shape (H, W)
    intercept : np.ndarray, shape (H, W)
    """
    if video.ndim != 3:
        raise ValueError("Input video must be a 3D array (T, H, W)")
    T, H, W = video.shape
    if T < 2:
        raise ValueError("Need at least 2 frames to fit a linear model")

    y_mat = video.reshape(T, -1)          # (T, H*W)
    t = np.arange(T, dtype=float).reshape(-1, 1)
    X = np.column_stack((t, np.ones(T)))  # (T, 2)

    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ X.T @ y_mat          # (2, H*W)

    slope = beta[0].reshape(H, W)
    intercept = beta[1].reshape(H, W)

    y_pred = X @ beta                     # (T, H*W)
    residuals = y_mat - y_pred
    res = residuals.reshape(T, H, W)

    return res, slope, intercept

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

def create_time_encoded_array(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True):
    """
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create frames with trails.
    Shorter windows make intuitive speed by trail length easier to see and work better for scans of densely populated plates.
    Longer windows are better for seeing behavioral movement patterns, as long as the plate isn't too densely populated.
    This implementation supports colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array of shape (N, 3) containing the colormap colors (B, G, R values 0-255), applied to each trail frame to encode temporal information.
        window: Integer value for the window size (number of frames to look back) used to create trails. Shorter windows take less processing time and enhance speed perception.
        scale_factor: Float value for the scaling factor, applied to the trails to adjust brightness. Higher values increase contrast. Default is 1.
        offset: Integer value for the brightness offset, positive values brighten the image, negative values darken it and can counteract noise. Default is 0.
        light_background: Boolean value. If True, assumes dark trails on light background and inverts the trail rendering. If False, assumes bright trails on dark background.

    Returns:
        time_encoded_array: 4D Numpy array of 8 bit unsigned integers (uint8) containing the time encoded array of frames with trails, with shape (time, height, width, 3), where axis 0 is time and axis 3 is color channels.

    Notes:
        Only returns arrays up to the last frame where a complete window fits.
    """
    
    time_encoded_array = []

    # loop through the average subtracted array and create time encoded frames for the windows starting at each frame
    for i in range(average_subtracted_array.shape[0]-window):
        time_encoded_frame = create_time_encoded_frame(average_subtracted_array, colormap=colormap, window=window, start_time=i, scale_factor=scale_factor, offset=offset, light_background=light_background)
        time_encoded_array.append(time_encoded_frame)

    return np.stack(time_encoded_array, axis=0)

def create_time_encoded_frame(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, start_time=0, scale_factor=1, offset=0, light_background=True):
    """
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create a single frame with trails.
    This implementation supports colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array of shape (N, 3) containing the colormap colors (B, G, R values 0-255), applied to the trail frame with color values being applied in order scaled to the window size.
        window: Integer value for the window size (number of frames to look back), used to create trails. Shorter windows take less processing time.
        start_time: Integer value for the start time of the frame. The window starts at this frame and continues forward for the specified window size.
        scale_factor: Float value for the scaling factor, applied to the trails to adjust brightness. Higher values increase contrast. Default is 1.
        offset: Integer value for the brightness offset, positive values brighten the image, negative values darken it and can counteract noise. Default is 0.
        light_background: Boolean value. If True, assumes dark trails on light background and inverts the trail rendering. If False, assumes bright trails on dark background.

    Returns:
        time_encoded_frame: 3D Numpy array of 8 bit unsigned integers (uint8) containing the time encoded frame with trails, with color channel as axis 3 (last axis), shape (height, width, 3).

    Notes:
        Uses numpy minimum/maximum projection to combine trail information from multiple frames.
    """
    time_encoded_frame = None

    # loop through the window and create colormapped frames
    for t in range(window):
        tmap = colormap[int(np.shape(colormap)[0]*t/window),:] # get the color for the current frame from the colormap
        if light_background: # invert the color used for colormapping if using a light background since it's going to be inverted back later
            tmap = 255 - tmap

        # set the color of each pixel based on the colormap and the brightness based on the corresponding average subtracted frame
        colormapped_frame = np.clip(cv2.merge([tmap[0]*average_subtracted_array[start_time+t]/255, tmap[1]*average_subtracted_array[start_time+t]/255, tmap[2]*average_subtracted_array[start_time+t]/255])*scale_factor + offset, 0, 255).astype(np.uint8)
        if light_background: # invert the colormapped frame if a light background is desired
            colormapped_frame = 255 - colormapped_frame

        # initialize or update the time encoded frame
        if time_encoded_frame is None:
            time_encoded_frame = colormapped_frame
        elif light_background:
            time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0) # minimum pixel value is used for projections if we want dark tracks on a light background
        else:
            time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0) # maximum pixel value is used for projections if we want light tracks on a dark background

    return time_encoded_frame

def add_timestamp(video_array, black_background=True, font_scale=1, font_thickness=1, seconds_per_frame=1, inPlace=False):
    """
    Adds a timestamp to each frame of the video array.
    This should be done last in the processing pipeline, after time encoding returns a color array.
    
    Args:
        video_array: 4D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with shape (time, height, width, 3), where time is axis 0 and color channels are the last axis.
        black_background: Boolean value. If True, uses white text on the assumption of a black background. If False, uses black text for a light background.
        font_scale: Float value for the OpenCV font scale. Controls the size of the timestamp text. Default is 1.
        font_thickness: Integer value for the OpenCV font thickness. Default is 1.
        seconds_per_frame: Float value for the frame time in seconds. This is the actual captured frame time (not the desired export frame time). Default is 1 second per frame.
        inPlace: Boolean value. If True, modifies the original array in place and returns None. If False (default), creates a copy and returns the copy.

    Returns:
        video_array: 4D Numpy array of 8 bit unsigned integers (uint8) containing the video frames with timestamps in bottom-left corner, with time as axis 0 and color channel as axis 3 (last axis). Returns None if inPlace=True.

    Notes:
        Only accepts color (3-channel) input since cv2.putText requires a color image.
        Timestamps are formatted as MM:SS and placed in the bottom-left corner of each frame.
    """
    if not inPlace:
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

    if not inPlace:
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