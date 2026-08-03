import cv2
import numpy as np
import pandas as pd

from wormtrails.quantitative import measure_window, measure_component, calculate_relative_metrics

def show_motion(
    video_path,
    worm_length = 10,
    scale_factor = 1,
    offset = 0,
):
    """
    Creates and shows scaled motion frames from a source video

    Args:
        video_path: String path to the video file.
        worm_length: Integer typical length of a worm in pixels.

    Returns:
        4D Numpy array (T, Y, X, C) of 8 bit unsigned integers (uint8) with time encoded by color.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """

    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    average_frame = get_average_frame(cap)

    for motion_frame in get_motion(
        cap,
        reference_frame=average_frame,
        kernel_radius=worm_length
    ):
        motion = motion_frame

        # Multiply motion by scale_factor and add offset to improve visibility
        motion *= scale_factor
        motion += offset

        # Clip and convert to 8-bit unsigned integer
        vis = np.clip(motion, 0, 255).astype(np.uint8)
        cv2.imshow("vis", vis)
        cv2.waitKey(1)

    # Release resources
    cv2.destroyAllWindows()
    cap.release()

def show_time_encoded_frame(
    video_path,
    worm_length,
    scale_factor = 1,
    offset = 0,
    start_frame = 0,
    window = None,
    colormap = np.array([[0,0,0]]),
    light_background=True,
):
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    average_frame = get_average_frame(cap)

    vis = get_time_encoded_frame(
        cap,
        reference_frame=average_frame, 
        kernel_radius = worm_length, 
        scale_factor = scale_factor,
        offset = offset,
        start_frame = start_frame,
        window = window,
        colormap = colormap,
        light_background=light_background,
    )

    cv2.imshow("vis", vis)
    cv2.waitKey(0)

    # Release resources
    cv2.destroyAllWindows()
    cap.release()

def measure_chemotaxis_streaming(
    video_path,
    thresh = 3,
    worm_length = 10,
    window=10, 
    interval=60, 
    minimum_size=10, 
    maximum_size=1000, 
    test_spot=None, 
    calibration=None
):
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    average_frame = get_average_frame(cap)

    worm_data = []

    for t in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - window + 1, interval):
        print(t)
        binary_array = []
        for motion_frame in get_motion(
            cap,
            reference_frame=average_frame,
            kernel_radius=worm_length,
            start_frame=t,
            window=window
        ):
            binary = np.zeros_like(motion_frame, dtype=np.uint8)
            binary[motion_frame > thresh] = 255
            binary_array.append(binary)
        
        binary_array = np.stack(binary_array, axis=0)
        window_metrics = measure_window(binary_array, window, minimum_size, maximum_size, calibration=calibration)

        for m in window_metrics:
            m['time'] = t
            # Calculate chemotaxis-specific metrics
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

    
def get_average_frame(
    cap
):
    """
    Calculates the average frame of a video given as a VideoCapture object

    Args:
        cap: OpenCV VideoCapture object for the video file to calculate the average frame of

    Returns:
        2D Numpy array (Y, X) of 64 bit floating point numbers (float64).
    """

    # Reset frame pointer position to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Calculate average frame
    sum_frame = None
    n_frames = 0
    while True:
        print(60*" ", end="\r")
        print(n_frames, end='\r')
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sum_frame is None:
            sum_frame = np.zeros_like(frame, dtype=np.float64)
        sum_frame += frame.astype(np.float64)
        n_frames += 1

        if n_frames == 0:
            raise ValueError("No frames read from video file")

    average_frame = sum_frame / n_frames

    return average_frame.astype(np.float64)


def get_motion(
    cap,
    reference_frame, 
    kernel_radius = 21, 
    start_frame = 0,
    window = None
    ):
    """
    Creates a motion frame from a raw frame and a reference frame

    Args:
        cap: OpenCV VideoCapture object for the video file
        reference_frame: 2D Numpy array (Y, X), typically average or max/min projection
        kernel_radius: Integer, brightness variations larger than this will be evened out

    Yields:
        2D Numpy array of 64 bit floating point numbers (float64) for each frame.

    Notes:
        The reference frame must have the same corrections performed as the 
    """
    reference_frame = reference_frame.copy().astype(np.float64)

    blur_frame = cv2.medianBlur(reference_frame.copy().astype(np.uint8), ksize=kernel_radius*2+1).astype(np.float64)
    target_brightness = np.mean(reference_frame).astype(np.float64)

    # Perform the same operations to the average frame that we will to each video frame since it's being used as stationary reference
    reference_frame *= target_brightness/blur_frame

    # Reset the frame pointer position to given start frame
    frame_number = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if window is not None:
            if frame_number >= start_frame + window:
                break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Normalize brightness
        frame *= target_brightness/np.mean(frame)

        # Even out large scale brightness variation (effective high pass filter)
        frame *= target_brightness/blur_frame

        # Calculate absolute difference from stationary reference frame
        motion = np.abs(frame - reference_frame)

        yield motion

        frame_number += 1

def get_time_encoded_frame(
    cap,
    reference_frame, 
    kernel_radius = 21, 
    scale_factor = 1,
    offset = 0,
    start_frame = 0,
    window = None,
    colormap = np.array([[0,0,0]]),
    light_background=True,
):
    time_encoded_frame = None
    t = 0

    for motion_frame in get_motion(
        cap,
        reference_frame=reference_frame,
        kernel_radius=kernel_radius,
        start_frame=start_frame,
        window=window
    ):
        motion = motion_frame

        # Multiply motion by scale_factor and add offset to improve visibility
        motion *= scale_factor
        motion += offset

        tmap = colormap[int(np.shape(colormap)[0]*t/window),:] # get the color for the current frame from the colormap
        if light_background: # invert the color used for colormapping if using a light background since it's going to be inverted back later
            tmap = 255 - tmap
        
        # set the color of each pixel based on the colormap and the brightness based on the corresponding average subtracted frame
        colormapped_frame = np.clip(cv2.merge([tmap[0]*motion/255, tmap[1]*motion/255, tmap[2]*motion/255]), 0, 255).astype(np.uint8)
        if light_background: # invert the colormapped frame if a light background is desired
            colormapped_frame = 255 - colormapped_frame

        # initialize or update the time encoded frame
        if time_encoded_frame is None:
            time_encoded_frame = colormapped_frame
        elif light_background:
            # minimum pixel value is used for projections if we want dark tracks on a light background
            time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0)
        else:
            # maximum pixel value is used for projections if we want light tracks on a dark background
            time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0)
        
        t += 1

    return time_encoded_frame

def create_time_encoded_array_streaming(
    video_path,
    save_path = None,
    worm_length = 10,
    scale_factor = 1,
    offset = 0,
    window = 1,
    colormap = np.array([[0,0,0]]),
    light_background=True,
    ):
    """
    Full pipeline function to create a time encoded array from a video file

    Args:
        video_path: String path to the video file.
        worm_length: Integer typical length of a worm in pixels.

    Returns:
        4D Numpy array (T, Y, X, C) of 8 bit unsigned integers (uint8) with time encoded by color.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """

    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        fps = 60
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Calculate average frame
    sum_frame = None
    n_frames = 0
    while True:
        print(60*" ", end="\r")
        print(n_frames, end='\r')
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sum_frame is None:
            sum_frame = np.zeros_like(frame, dtype=np.float64)
        sum_frame += frame.astype(np.float64)
        n_frames += 1

        if n_frames == 0:
            raise ValueError("No frames read from video file")
    
    average_frame = sum_frame / n_frames
    blur_frame = cv2.medianBlur(average_frame.copy().astype(np.uint8), ksize=worm_length*2+1).astype(np.float64)
    target_brightness = np.mean(average_frame).astype(np.float64)

    # Perform the same operations to the average frame that we will to each video frame since it's being used as stationary reference
    average_frame *= target_brightness/blur_frame

    for t_global in range(n_frames-window):
        # Reset the frame pointer position to first frame
        frame_number = t_global
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        time_encoded_frame = None
        # Loop through processing for each frame in the window
        for t in range(window):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Normalize brightness
            frame *= target_brightness/np.mean(frame)

            # Even out large scale brightness variation (effective high pass filter)
            frame *= target_brightness/blur_frame

            # Calculate absolute difference from stationary reference frame
            motion = np.abs(frame - average_frame)

            # Multiply motion by scale_factor and add offset to improve visibility
            motion *= scale_factor
            motion += offset

            tmap = colormap[int(np.shape(colormap)[0]*t/window),:] # get the color for the current frame from the colormap
            if light_background: # invert the color used for colormapping if using a light background since it's going to be inverted back later
                tmap = 255 - tmap
            
            # set the color of each pixel based on the colormap and the brightness based on the corresponding average subtracted frame
            colormapped_frame = np.clip(cv2.merge([tmap[0]*motion/255, tmap[1]*motion/255, tmap[2]*motion/255]), 0, 255).astype(np.uint8)
            if light_background: # invert the colormapped frame if a light background is desired
                colormapped_frame = 255 - colormapped_frame

            # initialize or update the time encoded frame
            if time_encoded_frame is None:
                time_encoded_frame = colormapped_frame
            elif light_background:
                # minimum pixel value is used for projections if we want dark tracks on a light background
                time_encoded_frame = np.min([time_encoded_frame, colormapped_frame], axis=0)
            else:
                # maximum pixel value is used for projections if we want light tracks on a dark background
                time_encoded_frame = np.max([time_encoded_frame, colormapped_frame], axis=0)

        cv2.imshow("vis", time_encoded_frame)
        cv2.waitKey(1)


    # Release resources
    cv2.destroyAllWindows()
    cap.release()

hsv_rainbow = []
for i in range(180):
    hsv = np.uint8([[[i, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    hsv_rainbow.append(np.array([bgr[0], bgr[1], bgr[2]], dtype=np.float64))
hsv_rainbow = np.array(hsv_rainbow, dtype=np.float64)

"""
show_motion(
    "/Users/dante/Desktop/datasets/scan1/raw_1.avi",
    worm_length = 10,
    scale_factor = 30,
    offset = -30
)

show_time_encoded_frame(
    "/Users/dante/Desktop/datasets/scan1/raw_1.avi",
    worm_length = 10,
    scale_factor = 30,
    offset = -30,
    start_frame = 0,
    window = 600,
    colormap = hsv_rainbow,
    light_background=False,
)
"""

data = measure_chemotaxis_streaming("/Users/dante/Desktop/datasets/scan1/raw_1.avi")
data.to_csv("./test_streaming.csv")