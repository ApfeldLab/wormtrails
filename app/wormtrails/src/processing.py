import cv2
import numpy as np
import math

def correct_vignetting(video_array, kernel_size=None, use_median_blur=False, inplace=False):
    """
    Corrects vignetting in a video array by normalizing each frame to have the same brightness as the average frame and dividing by a blur of the average frame.
    If a single frame (2D array) is provided, the frame brightness adjustment step is skipped.
    This step can be skipped if initial scan is evenly lit and the lens used does not vignette.
    Kernel size must be odd and smaller values will make smaller brightness variations get evened out.

    Args:
        video_array: 2D or 3D Numpy array containing the video frames. If 3D, time is axis 0.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame. If None, the kernel size will be calculated as one quarter the image width.
        inplace: Boolean value. If True, modifies the original array and returns it. If False (default), creates a copy.
        use_median_blur: Boolean value. If True, uses median blur instead of Gaussian blur for the vignetting correction.

    Returns:
        video_array: 2D or 3D Numpy array of 8 bit unsigned integers containing the corrected video frames.
    """
    if not inplace:
        video_array = video_array.copy()

    # Handle 2D and 3D arrays by wrapping 2D in a new axis for unified processing
    is_single_frame = video_array.ndim == 2
    if is_single_frame:
        video_array_3d = video_array[np.newaxis, ...]
    else:
        video_array_3d = video_array

    average_frame = np.mean(video_array_3d, axis=0)
    if kernel_size is None:
        kernel_size = int(average_frame.shape[0]/8) * 2 + 1 # Kernel size is one quarter the image width
    if use_median_blur:
        blur_frame = cv2.medianBlur(average_frame.astype(np.uint8), kernel_size)
    else:
        blur_frame = cv2.GaussianBlur(average_frame, (kernel_size,kernel_size), 0)
    target_brightness = np.mean(average_frame)

    # loop through and correct each frame
    for i in range(video_array_3d.shape[0]):
        frame = video_array_3d[i].astype(np.float32)
        
        if not is_single_frame:
            frame_brightness = np.mean(frame)
            if frame_brightness != 0:
                scaled_frame = frame * (target_brightness / frame_brightness) # scale each frame to have the same brightness as the average frame
            else:
                scaled_frame = frame  # Avoid division by zero
        else:
            scaled_frame = frame # Skip frame brightness adjustment for single frames

        # divide each frame by the blur of the average frame to correct vignetting or other large scale brightness variations
        video_array_3d[i] = (scaled_frame * target_brightness / blur_frame).astype(np.uint8)

    return video_array

def subtract_average(video_array, average_start=0, average_end=-1, use_absolute_difference=True, inplace=False):
    """
    Subtracts the average frame from each frame in the video array.
    This effectively only shows pixels which change in value over the course of the recording.
    A range of frames can be specified to calculate the average frame, and an absolute difference can be used instead of clipped subtraction.
    An ideal averaging range contains no stationary worms, as these will appear in negative space after subtraction.

    Args:
        video_array: 3D Numpy array containing the video frames, with time as axis 0.
        average_start: Integer value for the start index of the frame range used to calculate the average frame.
        average_end: Integer value for the end index of the frame range used to calculate the average frame.
        use_absolute_difference: Boolean value for whether to use absolute difference or not.
        inplace: Boolean value. If True, modifies the original array and returns it. If False (default), creates a copy.

    Returns:
        video_array: 3D Numpy array of 8 bit unsigned integers containing the subtracted video frames, with time as axis 0.
    """
    if not inplace:
        video_array = video_array.copy()

    if average_start == 0 and average_end == -1:
        average_frame = np.mean(video_array, axis=0)
    else:
        average_frame = np.mean(video_array[average_start:average_end,:,:], axis=0)

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
        else:
            video_array[i] = np.clip(scaled_frame - average_frame, 0, None).astype(np.uint8)

    if not inplace:
        return video_array

def create_track_array(average_subtracted_array, window):
    """
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create frames with trails.
    This is the simplest implementation of trails, and does not support colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array containing the average subtracted video frames, with time as axis 0.
        window: Integer value for the window size used to create trails.

    Returns:
        track_array: 3D Numpy array of 8 bit unsigned integers containing the track array, with time as axis 0.
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
        average_subtracted_array: 3D Numpy array containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array containing the colormap, applied to each trail frame.
        window: Integer value for the window size used to create trails, shorter windows take less processing time.
        scale_factor: Float value for the scale factor, higher values effectively increase contrast.
        offset: Integer value for the offset, positive values brighten the image, negative values darken it and can counteract noise.
        light_background: Boolean value for the light background.

    Returns:
        time_encoded_array: 4D Numpy array of 8 bit unsigned integers containing the time encoded array of frames with trails, with time as axis 0 and color channel as axis 3 (last axis).
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
        average_subtracted_array: 3D Numpy array containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array containing the colormap, applied to the trail frame with color values being applied in order scaled to the window size.
        window: Integer value for the window size used to create trails, shorter windows take less processing time. This is the number of frames after start_time to include in the trail.
        start_time: Integer value for the start time of the frame. The window starts at this frame and continues forward for the specified window size.
        scale_factor: Float value for the scale factor, higher values effectively increase contrast.
        offset: Integer value for the offset, positive values brighten the image, negative values darken it and can counteract noise.
        light_background: Boolean value for the light background.

    Returns:
        time_encoded_frame: 3D Numpy array of 8 bit unsigned integers containing the time encoded frame with trails with color channel as axis 3 (last axis).
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

def add_timestamp(video_array, black_background = True, font_scale=1, font_thickness=1, seconds_per_frame=1, inplace=False):
    """
    Adds a timestamp to each frame of the video array.
    This should be done last in the processing pipeline, after time encoding returns a color array.
    
    Args:
        video_array: 4D Numpy array containing the video frames, with time as axis 0 and color channel as axis 3 (last axis).
        black_background: Boolean value for whether a black background is used.
        font_scale: Float value for the font scale.
        font_thickness: Integer value for the font thickness.
        seconds_per_frame: Float value for the frame time in seconds. This is the captured frame time, not the desired export frame time.
        inplace: Boolean value. If True, modifies the original array and returns nothing. If False (default), creates a copy.
    
    Returns:
        video_array: 4D Numpy array of 8 bit unsigned integers containing the video frames with timestamps, with time as axis 0 and color channel as axis 3 (last axis).
    """
    if not inplace:
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

    if not inplace:
        return video_array

def normalize_array(video_array):
    """
    Normalizes the video array to a range of 0-255.
    This is not necessary for properly exposed recordings, but can be useful for high bit depth or underexposed recordings.
    
    Args:
        video_array: 3D Numpy array containing the video frames, with time as axis 0.

    Returns:
        normalized_array: 3D Numpy array of 8 bit unsigned integers containing the normalized video frames, with time as axis 0.
    """
    scale_factor = 255/np.max(video_array) # set one scale factor for all frames to avoid flickering
    normalized_array = np.empty_like(video_array, dtype=np.uint8)

    for i in range(video_array.shape[0]):
        frame = video_array[i].astype(np.float32) # use 32 bit float while scaling
        scaled_frame = frame * scale_factor

        normalized_array[i] = scaled_frame.astype(np.uint8)

    return normalized_array

def threshold_array(array, threshold, dark_objects=False, output_value=255, inplace=False):
    """
    Thresholds an array to create a binary array.
    Uses memory efficient operations to avoid int64 intermediates which can cause MemoryErrors on large arrays.
    
    Args:
        array: Numpy array to threshold.
        threshold: Numeric value for the threshold.
        dark_objects: Boolean value. If True, values below threshold become output_value (useful for dark worms on light background). 
                      If False, values above threshold become output_value (useful for light trails on dark background).
        output_value: The value to use for the "high" pixels (default 255). Use 1 for masks.
        inplace: Boolean value. If True, modifies the original array and returns it. Only works on uint8 or float32 arrays.
        
    Returns:
        binary_array: uint8 Numpy array of 0s and output_value.
    """
    if inplace:
        thresh_type = cv2.THRESH_BINARY_INV if dark_objects else cv2.THRESH_BINARY
        if array.ndim == 3:
            for i in range(array.shape[0]):
                cv2.threshold(array[i], threshold, output_value, thresh_type, dst=array[i])
        else:
            cv2.threshold(array, threshold, output_value, thresh_type, dst=array)

    else:
        if dark_objects:
            binary_array = (array < threshold).astype(np.uint8)
        else:
            binary_array = (array >= threshold).astype(np.uint8)
        
        if output_value != 1:
            binary_array *= output_value
        return binary_array
