import cv2
import numpy as np
from .processing import create_time_encoded_frame

def show_video_array(video_array):
    """
    Displays a video array in a window with a trackbar to scroll through frames.
    Press ESC to exit the window.

    Args:
        video_array: 3D Numpy array containing the video frames, with time as axis 0.
    """
    
    num_frames = video_array.shape[0]

    # Callback function for trackbar (does nothing but required by OpenCV)
    def on_trackbar(val):
        frame = video_array[val]
        cv2.imshow('esc to exit', frame)

    # Create a window and trackbar
    cv2.namedWindow('esc to exit', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', 'esc to exit', 0, num_frames - 1, on_trackbar)

    # Show the first frame initially
    cv2.imshow('esc to exit', video_array[0])

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

def show_frame(frame):
    """
    Displays a single frame in a window.
    Press ESC to exit the window.

    Args:
        frame: 2D Numpy array containing the frame.
    """
    
    cv2.namedWindow('esc to exit', cv2.WINDOW_NORMAL)
    cv2.imshow('esc to exit', frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

def show_time_encoding(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True):
    """
    Displays a time-encoded video array in a window with a trackbar to scroll through frames.
    Press ESC to exit the window.
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
    """
    
    num_frames = average_subtracted_array.shape[0]

    # Callback function for trackbar (does nothing but required by OpenCV)
    def on_trackbar(val):
        frame = create_time_encoded_frame(average_subtracted_array, colormap=colormap, window=window, start_time=val, scale_factor=scale_factor, offset=offset, light_background=light_background)
        cv2.imshow('esc to exit', frame)

    # Create a window and trackbar
    cv2.namedWindow('esc to exit', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', 'esc to exit', 0, num_frames - 1 - window, on_trackbar)

    # Show the first frame initially
    cv2.imshow('esc to exit', average_subtracted_array[0])

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
