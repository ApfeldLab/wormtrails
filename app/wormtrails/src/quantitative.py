import cv2
import numpy as np
import pandas as pd

def count_video(
    video_array,
    min_worm_area=10,
    max_worm_area=200,
    max_stationary_worm_length=20,
    motion_thresh=None,
    strict_motion_thresh=None,
    stationary_thresh_offset=4,
    return_vis=True
):
    # currently fixed parameters:
    edge_contrast_loDiff = 2
    edge_contrast_upDiff = 10
    kernel_size_small = 3
    kernel_size_medium = 11

    # load video
    video = video_array.copy()

    # create kernels which we'll use throughout the pipeline
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_small, kernel_size_small))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_medium, kernel_size_medium))

    # get max intensity projection and motion array
    max_proj = np.max(video, axis=0)
    min_proj = np.min(video, axis=0)
    motion_proj = max_proj.copy() - min_proj.copy()

    # create mask to remove edges
    plate_mask = max_proj.copy()
    seed_point = [plate_mask.shape[0]//2, plate_mask.shape[1]//2]
    cv2.floodFill(plate_mask, None, seed_point, 255, edge_contrast_loDiff, edge_contrast_upDiff)
    plate_mask[plate_mask < 255] = 0
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel_medium)

    # use masked motion projection to choose motion thresholds if none are given
    motion_proj[plate_mask == 0] = 0
    if strict_motion_thresh is None:
        strict_motion_thresh, _ = cv2.threshold(motion_proj.copy()[motion_proj > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if motion_thresh is None:
        motion_thresh, _ = cv2.threshold(motion_proj.copy()[(motion_proj > 0) & (motion_proj < strict_motion_thresh)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # threshold
    ret, strict_motion_mask = cv2.threshold(motion_proj.copy(), strict_motion_thresh, 255, cv2.THRESH_BINARY)
    ret, motion_mask = cv2.threshold(motion_proj.copy(), motion_thresh, 255, cv2.THRESH_BINARY)

    # remove small noise and expand
    strict_motion_mask = cv2.erode(strict_motion_mask, kernel_small)
    strict_motion_mask = cv2.dilate(strict_motion_mask, kernel_medium)
    motion_mask = cv2.erode(motion_mask, kernel_small)
    motion_mask = cv2.dilate(motion_mask, kernel_medium)

    # get stationary objects
    stationary = min_proj.copy()
    stationary = cv2.adaptiveThreshold(
        stationary,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        kernel_size_medium,
        stationary_thresh_offset
    )
    stationary[plate_mask == 0] = 0

    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
    n_roaming = 0
    if return_vis:
        vis = video.copy()
    for l in range(1, num_labels):
        trail_length = (stats[l, cv2.CC_STAT_WIDTH]**2 + stats[l, cv2.CC_STAT_HEIGHT]**2)**0.5
        if trail_length > max_stationary_worm_length + kernel_size_medium: # a worm in such a region will be roaming, and all its pixels will surpass motion_thresh
            motion_mask[labels == l] = 0 # remove analyzed trails of roaming worms from the motion mask
            strict_motion_mask[labels == l] = 0
            if return_vis:
                vis[:, labels == l] = 128
            label_counts = []
            for t in range(video.shape[0]):
                motion_frame = max_proj.copy() - video[t].copy()
                motion_frame[labels != l] = 0 # only look at the current label
                ret, motion_frame = cv2.threshold(motion_frame, motion_thresh, 255, cv2.THRESH_BINARY)
                num_labels_t, labels_t, stats_t, centroids_t = cv2.connectedComponentsWithStats(motion_frame, connectivity=8)
                areas = stats_t[:, cv2.CC_STAT_AREA]
                areas[0] = 0
                is_valid = (areas >= min_worm_area) & (areas <= max_worm_area)
                label_count_t = np.sum(is_valid)
                label_counts.append(label_count_t)
                if return_vis:
                    vis[t, is_valid[labels_t]] = 255
            label_count = np.max(label_counts)
            n_roaming += label_count
            print(f"Label {l}: {label_count}  ", end="\r")
    print(" "*60, end="\r")

    # floodfill the stationary binary image from points in the motion mask
    alive_stationary = np.zeros_like(stationary)
    num_labels_sw, labels_sw, stats_sw, _ = cv2.connectedComponentsWithStats(stationary, connectivity=8)

    # Find labels in stationary that intersect with the remaining motion_mask
    overlapping_labels = np.unique(labels_sw[strict_motion_mask > 0])
    n_stationary_alive = 0
    for label_idx in overlapping_labels:
        if label_idx == 0: 
            continue
        elif stats_sw[label_idx, cv2.CC_STAT_AREA] >= min_worm_area and stats_sw[label_idx, cv2.CC_STAT_AREA] <= max_worm_area:
            alive_stationary[labels_sw == label_idx] = 255
            n_stationary_alive += 1

    if return_vis:
        vis[:, alive_stationary > 0] = 255
        video = vis

    return n_roaming, n_stationary_alive, video

def find_worms(
    video_array,
    plate_mask,
    min_size=10,
    max_size=300,
    corrected_thresh=None,
    strict_corrected_thresh=None,
    motion_thresh=None,
    strict_motion_thresh=None,
    kernel_size=None,
    high_sensitivity=False,
    inPlace=False
):
    """
    Finds living worms in a video array using motion detection and size filtering.
    Currently optimized for bright field illumination with a bright background.
    Recordings of 30 seconds to 1 minute are recommended for reliable results.
    
    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        plate_mask: 2D Numpy array of 8 bit unsigned integers (uint8) containing the plate mask. Plates should have pixel values greater than 0
        min_size: Integer value for the minimum size (pixel area) of a potential worm. Default is 10.
        max_size: Integer value for the maximum size (pixel area) of a potential worm. Default is 300.
        corrected_thresh: Integer value for the threshold for converting the video array to a binary array, with thresholded pixels being eroded before inclusion. If None (default), set to one less than the median pixel value.
        strict_corrected_thresh: Integer value for the threshold for converting the video array to a binary array, with all thresholded pixels included. If None (default), set to one less than corrected_thresh.
        motion_thresh: Integer value for the motion detection threshold. Pixels with motion values above this are considered moving, with thresholded pixels being eroded before inclusion. Default is Otsu's threshold of nonzero motion pixel values.
        strict_motion_thresh: Integer value for the motion detection threshold. Pixels with motion values above this are considered moving, with all thresholded pixels included. Default is motion_thresh plus one.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction. Default is double the maximum worm width, assuming a worm has an aspect ratio of 1:10.
        high_sensitivity: Boolean value. If True, the motion threshold will be allowed to be 0 if pixels are grouped together. False by default.
        inPlace: Boolean value. If True, the video array will be modified in place. If False (default), a copy of the video array will be used.

    Returns:
        3D Numpy array of 8 bit unsigned integers (uint8) the same shape as video_array. Living worms are 255 and background is 0.

    Raises:
        ValueError: If the video array cannot be processed or thresholds fail.

    Notes:
        - Uses vignetting correction with kernel-based blur
        - Detects potential objects with loose thresholds applied to the vignetting corrected and motion arrays, followed by erosion, and strict thresholds
        - Validates objects by requiring motion detection in each object's pixels
    """
    if not inPlace:
        video_array = video_array.copy()
    if kernel_size is None:
        kernel_size = int(np.sqrt(max_size / 10)) * 2 + 1

    motion = np.zeros_like(video_array)
    reference_frame = np.max(video_array, axis=0)
    blur_frame = cv2.medianBlur(reference_frame, kernel_size)
    target_brightness = np.mean(reference_frame)
    for i in range(video_array.shape[0]):
        frame = video_array[i].astype(np.float32)
        frame_brightness = np.mean(frame)
        if frame_brightness > 0:
            frame *= (target_brightness / frame_brightness)
        motion[i] = np.abs(frame.copy() - reference_frame).astype(np.uint8)
        video_array[i] = (frame * target_brightness / blur_frame).astype(np.uint8)
    
    motion[:, plate_mask == 0] = 0

    if corrected_thresh is None:
        corrected_thresh = np.median(video_array[:, plate_mask > 0]) - 1
    if motion_thresh is None:
        motion_thresh, _ = cv2.threshold(motion[motion > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if strict_corrected_thresh is None:
        strict_corrected_thresh = corrected_thresh - 1
    if strict_motion_thresh is None:
        strict_motion_thresh = motion_thresh + 1

    worms = np.zeros_like(video_array)
    # loose threshold
    worms[video_array < corrected_thresh] = 1
    worms[motion > motion_thresh] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    for t in range(worms.shape[0]):
        cv2.morphologyEx(worms[t], cv2.MORPH_OPEN, kernel, worms[t])

    # strict threshold
    worms[video_array < strict_corrected_thresh] = 1
    worms[motion > strict_motion_thresh] = 1
    worms[:, plate_mask == 0] = 0

    for t in range(worms.shape[0]):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(worms[t], connectivity=8)

        areas = stats[:, cv2.CC_STAT_AREA]

        moving_labels = labels.copy()
        if high_sensitivity:
            motion_binary = cv2.morphologyEx((motion[t] > motion_thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            motion_binary[motion[t] > strict_motion_thresh] = 1
            moving_labels[motion_binary == 0] = 0
        else:
            moving_labels[motion[t] < strict_motion_thresh] = 0

        moving_areas = np.bincount(moving_labels.ravel(), minlength=num_labels)

        is_alive = (areas >= min_size) & (areas <= max_size) & (moving_areas > 0)
        is_alive[0] = False
        is_small = (areas < min_size) & (moving_areas > 0)
        is_small[0] = False
        alive_worms = is_alive[labels]
        worms[t, alive_worms] = 255
        worms[t] |= cv2.dilate(is_small[labels].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(np.sqrt(30 / np.pi)) * 2 - 1,)*2)) * 255

    worms[worms < 255] = 0

    return worms

def calculate_relative_metrics(position, direction, test_spot):
    """
    Calculates polar coordinates and relative angle to a test spot for chemotaxis analysis.
    
    Args:
        position: Numpy array of shape (2,) containing (y, x) coordinates of the object.
        direction: Numpy array of shape (2,) containing (dy, dx) normalized direction vector.
        test_spot: Numpy array or tuple of (y, x) absolute coordinates for the test spot or bait location.
        
    Returns:
        r: Radial distance from test_spot to position (scalar)
        theta: Absolute angle in radians from -pi to pi
        relative_angle: Relative angle in radians from -pi to pi, where 0 faces directly away from test_spot.
        
    Notes:
        - Uses polar coordinate transformation for chemotaxis metric calculation
        - Relative angle is computed as the difference between direction vector and position vector to test_spot
    """
    rel_pos = position - np.array(test_spot)
    r = np.linalg.norm(rel_pos)
    theta = np.arctan2(rel_pos[0], rel_pos[1])
    
    # Relative angle: 0 faces directly away from test_spot
    rel_angle = np.arctan2(direction[0], direction[1]) - theta
    rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
    
    return r, theta, rel_angle

def measure_component(binary_window, component_mask, centroid_yx, time_window):
    """
    Measures movement metrics (position, direction, speed) for a single component in a time window.
    
    Args:
        binary_window: 3D Numpy array of 8 bit unsigned integers (uint8) for the time window, shape (time, height, width).
        component_mask: 2D mask (0 and 255) for the component in the projection.
        centroid_yx: Numpy array of shape (2,) containing (y, x) centroid from the projection.
        time_window: Number of frames in the window used for speed calculation.
        
    Returns:
        Dictionary with movement metrics for the component, or None if insufficient points to calculate metrics.
        Dictionary keys: 'y', 'x', 'direction_y', 'direction_x', 'speed'
        
    Notes:
        - Direction is determined by fitting a line to the 2D footprint (main axis)
        - Direction is resolved using temporal information to remove motion ambiguity
        - Speed is calculated as 2*(trail_radius - worm_radius) / time_window
    """
    component_points_2d = np.transpose(np.nonzero(component_mask))
    if len(component_points_2d) < 2:
        return None

    # Determine direction by fitting a line to the 2D footprint (main axis)
    position = centroid_yx
    vx, vy, _, _ = cv2.fitLine(component_points_2d[:, ::-1].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vy[0]), float(vx[0])]) # (dy, dx)

    # Resolve direction of motion ambiguity using temporal information
    points_3d = np.transpose(np.nonzero((binary_window > 0) & component_mask))
    
    if len(points_3d) > 0:
        mid_idx = time_window / 2
        early_pts = points_3d[points_3d[:, 0] < mid_idx]
        late_pts = points_3d[points_3d[:, 0] >= mid_idx]
        
        if len(early_pts) > 0 and len(late_pts) > 0:
            v_temporal = np.mean(late_pts[:, 1:], axis=0) - np.mean(early_pts[:, 1:], axis=0)
            if np.dot(direction, v_temporal) < 0:
                direction = -direction

    # Calculate speed: 2*(trail_radius - worm_radius) / time_window
    _, trail_radius = cv2.minEnclosingCircle(component_points_2d[:, ::-1].astype(np.float32))
    points_at_mid = points_3d[points_3d[:, 0] == int(time_window / 2)]
    if len(points_at_mid) > 0:
        _, worm_radius = cv2.minEnclosingCircle(points_at_mid[:, 1:][:, ::-1].astype(np.float32))
    else:
        worm_radius = trail_radius / 2

    speed = 2 * (trail_radius - worm_radius) / time_window

    return {
        'y': position[0],
        'x': position[1],
        'direction_y': direction[0],
        'direction_x': direction[1],
        'speed': speed
    }

def measure_window(binary_window, time_window, minimum_size=10, maximum_size=1000):
    """
    Labels connected components in a binary time window and measures movement for each.
    
    Args:
        binary_window: 3D Numpy array of 8 bit unsigned integers (uint8) for the time window, shape (time, height, width).
        time_window: Number of frames in the window used for analysis.
        minimum_size: Minimum pixel area in 2D projection for a component to be considered. Default is 10.
        maximum_size: Maximum pixel area in 2D projection for a component to be considered. Default is 1000.
        
    Returns:
        List of dictionaries containing movement metrics for each detected component.
        Each dictionary contains: 'y', 'x', 'direction_y', 'direction_x', 'speed', 'label_id', 'time'
        
    Notes:
        - Components are identified using connected components analysis on max projection
        - Components outside the size range are filtered out
        - Only components with sufficient data points are included in results
    """
    projected = np.max(binary_window, axis=0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(projected.astype(np.uint8), connectivity=8)
    
    window_data = []
    for label_id in range(1, num_labels):
        if minimum_size < stats[label_id, cv2.CC_STAT_AREA] < maximum_size:
            comp_mask = (labels == label_id)
            centroid_yx = centroids[label_id][::-1]
            metrics = measure_component(binary_window, comp_mask, centroid_yx, time_window)
            if metrics:
                metrics['label_id'] = label_id
                window_data.append(metrics)
    return window_data

def measure_chemotaxis(binary_array, time_window=10, interval=60, minimum_size=10, maximum_size=1000, test_spot=None):
    """
    Measures chemotaxis metrics over time windows using a 2D projection approach.
    Analyzes binary video data to compute position, direction, speed, and relative angle to a test spot.
    
    Args:
        binary_array: 3D Numpy array of shape (time, height, width) containing 8-bit unsigned integers (0/255 binary).
        time_window: Number of frames in each sliding window to analyze. Default is 10.
        interval: Frame interval between the start of consecutive time windows. Default is 60 (1 minute).
        minimum_size: Minimum area in pixels (2D projection) to consider a component as a valid worm trail. Default is 10.
        maximum_size: Maximum area in pixels (2D projection) to consider a component as a valid worm trail. Default is 1000.
        test_spot: Tuple or array of (y, x) absolute coordinates for the test spot or bait location. If None (default), relative angle metrics are not computed.
        
    Returns:
        A pandas DataFrame with one row per detected worm trail, containing columns:
            - 'y', 'x': Position of the component
            - 'direction_y', 'direction_x': Direction vector components
            - 'speed': Calculated speed in the window
            - 'time': Starting frame index of the time window
            - 'r': Radial distance to test spot (if test_spot provided)
            - 'theta': Absolute angle in radians
            - 'relative_angle': Relative angle to test spot in radians
            
        Prints progress indicator during processing.

    Notes:
        - Uses sliding window approach with specified overlap
        - Progress is printed to stdout with frame count indicator
    """
    worm_data = []
    
    # Iterate for consecutive time windows
    for t in range(0, binary_array.shape[0] - time_window + 1, interval):
        print(f'{t}/{binary_array.shape[0]}', end="\r")
        
        window_metrics = measure_window(binary_array[t:t+time_window], time_window, minimum_size, maximum_size)
        
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
            
            worm_data.append(m)

    return pd.DataFrame(worm_data)

