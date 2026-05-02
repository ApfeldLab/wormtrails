import cv2
import numpy as np
from .processing import create_time_encoded_frame, fit_pixel_linear_model

def show_video_array(video_array, window_name='esc to exit'):
    """
    Displays a video array in a window with a trackbar to scroll through frames.
    Close the window to exit.

    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.

    Returns:
        None. Displays video in an OpenCV window.

    Notes:
        - Uses OpenCV's imshow and createTrackbar for frame navigation
        - Close the window to exit
    """
    
    num_frames = video_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - 1, lambda x: None)

    # stop and reinitialize because otherwise there are errors if closing via GUI on first run
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - 1, lambda x: None)
    
    while True:
        current_idx = cv2.getTrackbarPos('Frame', window_name)
        frame = video_array[current_idx]

        if frame.dtype in (np.float32, np.float64):
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        cv2.imshow(window_name, frame)
        
        # 1. Primary exit: ESC or 'q' (most reliable across all environments)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # 2. Secondary exit: Window 'X' button (with explicit cleanup)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # Explicitly destroy to prevent QT backend from keeping it alive
            cv2.destroyWindow(window_name)
            break

    # Flush remaining events and clean up
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def show_frame(frame, window_name='esc to exit'):
    """
    Displays a single frame in a window.
    Close the window to exit.

    Args:
        frame: 2D Numpy array of 8 bit unsigned integers (uint8) containing a single frame.

    Returns:
        None. Displays frame in an OpenCV window.

    Notes:
        - Uses OpenCV's imshow for single frame display
        - Close the window to exit
    """
    
    # stop and reinitialize because otherwise there are errors if closing via GUI on first run
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    cv2.imshow(window_name, frame)
    
    while True:
        try:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if visible <= 0:
                break
        except cv2.error:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyWindow(window_name)

def show_time_encoding(average_subtracted_array, colormap=np.array([[0,0,0]]), window=1, scale_factor=1, offset=0, light_background=True, window_name='esc to exit'):
    """
    Displays a time-encoded video array in a window with a trackbar to scroll through frames.
    Close the window to exit.
    Projects along the time axis of the average subtracted array within a local frame window of the specified size to create frames with trails.
    Shorter windows make intuitive speed by trail length easier to see and work better for scans of densely populated plates.
    Longer windows are better for seeing behavioral movement patterns, as long as the plate isn't too densely populated.
    This implementation supports colormaps to encode time within frames.
    
    Args:
        average_subtracted_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the average subtracted video frames, with time as axis 0.
        colormap: Numpy array of shape (N, 3) containing the colormap colors (B, G, R values 0-255), applied to each trail frame.
        window: Integer value for the window size (number of frames to look back) used to create trails. Shorter windows take less processing time.
        scale_factor: Float value for the scaling factor, applied to trails to adjust brightness. Higher values increase contrast.
        offset: Integer value for the brightness offset, positive values brighten the image, negative values darken it and can counteract noise.
        light_background: Boolean value. If True, assumes dark trails on light background. If False, assumes bright trails on dark background.
    """
    
    num_frames = average_subtracted_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - window, lambda x: None)

    # stop and reinitialize because otherwise there are errors if closing via GUI on first run
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - window, lambda x: None)
    
    while True:
        current_idx = cv2.getTrackbarPos('Frame', window_name)
        frame = frame = create_time_encoded_frame(average_subtracted_array, colormap=colormap, window=window, start_time=current_idx, scale_factor=scale_factor, offset=offset, light_background=light_background)

        if frame.dtype in (np.float32, np.float64):
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        cv2.imshow(window_name, frame)
        
        # 1. Primary exit: ESC or 'q' (most reliable across all environments)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # 2. Secondary exit: Window 'X' button (with explicit cleanup)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # Explicitly destroy to prevent QT backend from keeping it alive
            cv2.destroyWindow(window_name)
            break

    # Flush remaining events and clean up
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def count_assist(video_array, window_name='count assist'):
    """
    Displays a video overlaid with motion trails, allowing the user to mark worms 
    by double-clicking. Use the backspace key to undo a marker.
    
    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames.
        window_name: Title of the display window (e.g., file name).
        
    Returns:
        markers: List of (x, y) coordinate tuples for marked worms.
    """
    num_frames = video_array.shape[0]
    if num_frames == 0:
        print("Warning: Empty video array.")
        return

    # Calculate the motion overlay
    overlay_video = []
    for t in range(2, num_frames):
        rss_t, _, _ = fit_pixel_linear_model(video_array[:t])
        motion = rss_t/t
        motion[motion < 1] = 1
        log_motion = np.log2(motion.astype(np.float64))
        log_motion *= 255 / np.max(log_motion)
        log_motion = np.clip(log_motion, 0, 255).astype(np.uint8)
        overlay_video.append(video_array[t] // 2 + log_motion // 2)

    markers = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - 3, lambda x: None)

    # Initialize window properly
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', window_name, 0, num_frames - 3, lambda x: None)

    state = {'current_idx': 0, 'needs_redraw': True}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            markers.append((x, y))
            print(f"Total marked worms: {len(markers)}", end='\r')
            state['needs_redraw'] = True

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        idx = cv2.getTrackbarPos('Frame', window_name)
        if idx != state['current_idx']:
            state['current_idx'] = idx
            state['needs_redraw'] = True

        if state['needs_redraw']:
            frame = overlay_video[state['current_idx']]
            if frame.dtype in (np.float32, np.float64):
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Convert to BGR to draw red dots
            if len(frame.shape) == 2:
                color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                color_frame = frame.copy()
                
            for (mx, my) in markers:
                cv2.circle(color_frame, (mx, my), 3, (0, 0, 255), -1)

            cv2.imshow(window_name, color_frame)
            state['needs_redraw'] = False

        key = cv2.waitKey(30) & 0xFF
        
        # Primary exit: ESC or 'q'
        if key in (27, ord('q')):
            break
            
        # Undo: Backspace (8) or Delete (127)
        if key in (8, 127):
            if markers:
                markers.pop()
                print(f"Total marked worms: {len(markers)}", end='\r')
                state['needs_redraw'] = True

        # Secondary exit: Window 'X' button
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            cv2.destroyWindow(window_name)
            break

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return markers