import cv2
import numpy as np

def read_video_file(video_path):
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
    num_frames = video_array.shape[0]
    height = video_array.shape[1] 
    width = video_array.shape[2]
    if len(video_array.shape) > 3:
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