import cv2
import numpy as np
import math

def correct_vignetting(video_array):
    average_frame = np.mean(video_array, axis=0)
    kernel_size = int(average_frame.shape[0]/8) * 2 + 1
    blur_frame = cv2.GaussianBlur(average_frame, (kernel_size,kernel_size), 0)
    target_brightness = np.mean(average_frame)

    for i in range(video_array.shape[0]):
            frame = video_array[i].astype(np.float32)
            frame_brightness = np.mean(frame)

            # Avoid division by zero
            if frame_brightness != 0:
                scaled_frame = frame * (target_brightness / frame_brightness)
            else:
                scaled_frame = frame  # Leave unchanged if average is zero

            video_array[i] = (scaled_frame * target_brightness / blur_frame).astype(np.uint8)

    return video_array

def subtract_average(video_array, average_start=0, average_end=-1, use_absolute_difference=True):
    if average_start == 0 and average_end == -1:
        average_frame = np.mean(video_array, axis=0)
    else:
        average_frame = np.mean(video_array[average_start:average_end,:,:], axis=0)

    target_brightness = np.mean(average_frame)

    for i in range(video_array.shape[0]):
            frame = video_array[i].astype(np.float32)
            frame_brightness = np.mean(frame)

            # Avoid division by zero
            if frame_brightness != 0:
                scaled_frame = frame * (target_brightness / frame_brightness)
            else:
                scaled_frame = frame  # Leave unchanged if average is zero

            if use_absolute_difference:
                video_array[i] = np.abs(scaled_frame - average_frame).astype(np.uint8)
            else:
                video_array[i] = np.clip(scaled_frame - average_frame, 0, None).astype(np.uint8)

    return video_array

def create_track_array(average_subtracted_array, window):
    track_array = []
    for i in range(average_subtracted_array.shape[0]):
        track_frame = np.max(average_subtracted_array[i:i+window,:,:], axis=0)
        track_array.append(track_frame)

    return np.stack(track_array, axis=0)

def create_time_encoded_array(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, scale_factor=1, offset=0, light_background=True):
    time_encoded_array = []
    for i in range(average_subtracted_array.shape[0]-window):
        time_encoded_frame = create_time_encoded_frame(average_subtracted_array, colormap=colormap, window=window, start_time=i, scale_factor=scale_factor, offset=offset, light_background=light_background)
        time_encoded_array.append(time_encoded_frame)

    return np.stack(time_encoded_array, axis=0)

def create_time_encoded_frame(average_subtracted_array, colormap=np.array([[0,0,0]]), window=20, start_time=0, scale_factor=1, offset=0, light_background=True):
    mono_array = average_subtracted_array[start_time:start_time+window,:,:]
    colormapped_array = []
    for t in range(window):
        tmap = colormap[int(np.shape(colormap)[0]*t/window),:]
        if light_background:
            tmap = 255 - tmap
        colormapped_frame = np.clip(cv2.merge([tmap[0]*mono_array[t]/255, tmap[1]*mono_array[t]/255, tmap[2]*mono_array[t]/255])*scale_factor + offset, 0, 255).astype(np.uint8)
        if light_background:
            colormapped_frame = 255 - colormapped_frame
        colormapped_array.append(colormapped_frame)

    if light_background:
        return np.min(colormapped_array, axis=0)
    else:
        return np.max(colormapped_array, axis=0)

def add_timestamp(video_array, black_background = True, font_scale=1, font_thickness=1, seconds_per_frame=1):
    num_frames, height, width = video_array.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, height-10)
    if black_background:
        color = 255 # 255 for white text, 0 for black text
    else:
        color = 0

    for i in range(num_frames):
        total_seconds = int(i*seconds_per_frame)
        timestamp = f"{(total_seconds // 60):02d}:{(total_seconds % 60):02d}"
        color_frame = cv2.cvtColor(video_array[i].copy(), cv2.COLOR_GRAY2BGR)
        cv2.putText(color_frame, timestamp, position, font, font_scale, (color, color, color), font_thickness)
        video_array[i] = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    return video_array

def normalize_array(video_array):
    scale_factor = 255/np.max(video_array)
    normalized_array = np.empty_like(video_array, dtype=np.uint8)

    for i in range(video_array.shape[0]):
        frame = video_array[i].astype(np.float32)
        scaled_frame = frame * scale_factor

        normalized_array[i] = scaled_frame.astype(np.uint8)

    return normalized_array