import cv2
import numpy as np

def read_video_file(video_path):
    """
    Reads a video file and returns a 3D Numpy array containing the video frames, with time as axis 0.
    Only monochrome videos are supported, and color videos are automatically converted to greyscale.

    Args:
        video_path: String path to the video file.

    Returns:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.

    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Convert frame to greyscale
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grey_frame.astype(np.uint8))
        
    cap.release()
        
    # Convert list of frames to a 3D NumPy array
    video_array = np.stack(frames, axis=0)
    return video_array

def write_mp4(video_array, out_path, fps=60):
    """
    Writes a 3D Numpy array containing video frames to an MP4 file.
    
    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        out_path: String path to the output MP4 file.
        fps: Integer value for the frames per second to write the video at. Default is 60.

    Prints:
        Confirmation message when video writing completes.
    """
    
    num_frames = video_array.shape[0]
    height = video_array.shape[1] 
    width = video_array.shape[2]
    if len(video_array.shape) > 3: # allow for color or monochrome input videos
        isColor = True
    else:
        isColor = False

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=isColor)

    for i in range(num_frames):
        frame = video_array[i]

        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Write the frame
        out.write(frame)

    out.release()
    print(f"Video successfully written to {out_path}")

def write_avi(video_array, out_path, fps=60):
    """
    Writes a 3D Numpy array containing video frames to an AVI file using FFV1 codec for lossless compression.

    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        out_path: String path to the output AVI file.
        fps: Integer value for the frames per second to write the video at. Default is 60.

    Prints:
        Confirmation message when video writing completes.
    """
    
    num_frames = video_array.shape[0]
    height = video_array.shape[1] 
    width = video_array.shape[2]
    if len(video_array.shape) > 3:
        isColor = True
    else:
        isColor = False

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Codec for lossless .avi
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=isColor)

    for i in range(num_frames):
        frame = video_array[i]

        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Write the frame
        out.write(frame)

    out.release()
    print(f"Video successfully written to {out_path}")
