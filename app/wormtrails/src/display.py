import cv2
import numpy as np
from .processing import create_time_encoded_frame

def show_video_array(video_array):
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
    cv2.namedWindow('esc to exit', cv2.WINDOW_NORMAL)
    cv2.imshow('esc to exit', frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

def show_time_encoding(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True):
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
