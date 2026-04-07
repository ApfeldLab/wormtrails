import cv2
import numpy as np
import pandas as pd

def count_video(
    video_array, 
    min_size=10, 
    max_size=300, 
    persistence=1,
    corrected_thresh=None, 
    motion_thresh=3, 
    kernel_size=11,  
    plate_edge_size=None, 
    plate_width=None,
    high_sensitivity=False,
    detailed_output=False,
    inPlace=False
):
    """
    Counts the number of living worms in a video array using motion detection and size filtering.
    Currently optimized for bright field illumination with a bright background.
    Recordings of 30 seconds to 1 minute are recommended for reliable results.
    
    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        min_size: Integer value for the minimum size (pixel area) of a potential worm. Default is 10.
        max_size: Integer value for the maximum size (pixel area) of a potential worm. Default is 300.
        persistence: Integer value for the number of frames a worm must be detected in to be counted. Default is 1.
        corrected_thresh: Integer value for the threshold for converting the video array to a binary array. If None (default), set to one less than the median pixel value.
        motion_thresh: Integer value for the motion detection threshold. Pixels with motion values above this are considered moving. Default is the 99.9th percentile of motion pixel values.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction. Default is 11.
        plate_edge_size: Integer value for the width in pixels of the plate edge to be excluded when mask_plate=True. Default is 20% the width of plate_width.
        plate_width: Integer value for the width of the plate in pixels. Default is 80% the width of the frame.
        high_sensitivity: Boolean value. If True, the motion threshold will be allowed to be 0 if pixels are grouped together. False by default.
        detailed_output: Boolean value. If False (default), returns only the count. If True, returns a tuple with count, and visualization.
        inPlace: Boolean value. If True, the video array will be modified in place. If False (default), a copy of the video array will be used.

    Returns:
        If detailed_output is False:
            An integer value for the number of living worms detected.
        If detailed_output is True:
            A tuple containing (count, visualization) where:
                - count: Integer count of validated living worms
                - labeled_worms: 3D Numpy array of 16 bit unsigned integers (uint16) the same shape as video_array. Each worm is assigned a unique integer value.

    Raises:
        ValueError: If the video array cannot be processed or thresholds fail.

    Notes:
        - Uses vignetting correction with kernel-based blur
        - Applies plate masking when enabled to avoid edge artifacts
        - Validates objects by requiring motion detection in each object's pixels
    """
    if not inPlace:
        video_array = video_array.copy()
    
    plate_mask = create_plate_mask(np.mean(video_array, axis=0), edge_size=plate_edge_size, plate_width=plate_width)

    worms = find_worms(
        video_array, 
        plate_mask, 
        min_size=min_size, 
        max_size=max_size, 
        corrected_thresh=corrected_thresh, 
        motion_thresh=motion_thresh, 
        kernel_size=kernel_size, 
        high_sensitivity=high_sensitivity,
        inPlace=True)
    labeled_worms = track_and_label_worms(worms, persistence=persistence)

    count = len(np.unique(labeled_worms)) - 1

    if detailed_output:
        return count, labeled_worms
    else:
        return count

def find_worms(
    video_array,
    plate_mask,
    min_size=10,
    max_size=300,
    corrected_thresh=None,
    motion_thresh=None,
    kernel_size=11,
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
        corrected_thresh: Integer value for the threshold for converting the video array to a binary array. If None (default), set to one less than the median pixel value.
        motion_thresh: Integer value for the motion detection threshold. Pixels with motion values above this are considered moving. Default is the 99.9th percentile of motion pixel values.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction. Default is 11.
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
    
    if corrected_thresh is None:
        corrected_thresh = np.median(video_array[:, plate_mask > 0]) - 1
    if motion_thresh is None:
        motion_thresh = np.quantile(motion[:, plate_mask > 0], 0.999)

    worms = np.zeros_like(video_array)
    # loose threshold
    worms[video_array < corrected_thresh] = 1
    worms[motion > 0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    for t in range(worms.shape[0]):
        cv2.morphologyEx(worms[t], cv2.MORPH_OPEN, kernel, worms[t])

    # strict threshold
    worms[video_array < corrected_thresh - 1] = 1
    worms[motion > motion_thresh] = 1
    worms[:, plate_mask == 0] = 0

    for t in range(worms.shape[0]):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(worms[t], connectivity=8)

        areas = stats[:, cv2.CC_STAT_AREA]

        moving_labels = labels.copy()
        if high_sensitivity:
            motion_binary = cv2.morphologyEx((motion[t] > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            motion_binary[motion[t] > motion_thresh] = 1
            moving_labels[motion_binary == 0] = 0
        else:
            moving_labels[motion[t] < motion_thresh] = 0

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

def create_plate_mask(frame, edge_size=None, plate_width=None, light_background=True):
    """
    Creates a mask for the plate from a raw (uncorrected) still frame.
    Determines the plate region by fitting circles to objects in a downsampled version of the image.
    Optimized for speed and memory usage.
    
    Args:
        frame: 2D Numpy array of 8 bit unsigned integers (uint8) containing a single raw (uncorrected) video frame.
        edge_size: Integer value for the size of the edge of the plate to be excluded in pixels. Default is 20% the width of plate_width.
        plate_width: Integer value for the width of the plate in pixels. Default is 80% the width of the frame.
        light_background: Boolean value. If True, assumes dark objects on light background. If False, assumes bright objects on dark background.

    Returns:
        A 2D Numpy array of 8 bit unsigned integers (uint8) containing the mask for the plate, with 1 for the plate region and 0 for the background.

    Notes:
        - Downsamples the frame to a maximum dimension of 1024 for efficient processing
        - Uses local thresholding (block size ~1/8 width) followed by contour analysis to identify circular objects
        - Fits a circle to each candidate plate and applies the requested edge margin
        - Returns a full-resolution mask
    """

    # Ensure input is uint8 for OpenCV operations
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Downsample for speed and memory efficiency
    h, w = frame.shape
    target_dim = 1024
    if max(h, w) > target_dim:
        scale = target_dim / max(h, w)
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small_frame = frame

    if plate_width is None:
        plate_width = max(h, w) * 0.8
    if edge_size is None:
        edge_size = int(plate_width * 0.2)
    plate_width *= scale

    # Pre-process to reduce noise and small artifacts
    small_blurred = cv2.medianBlur(small_frame, 5)

    # Use local thresholding to identify the plate
    # block_size should be around 1/8 of the image width and must be odd
    block_size = int(plate_width) * 2 + 1

    if not light_background:
        binary = cv2.adaptiveThreshold(small_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, block_size, 0)
    else:
        binary = cv2.adaptiveThreshold(small_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, block_size, 0)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify large circular contours that likely represent plates
    min_area = (small_frame.shape[0] * small_frame.shape[1]) * 0.05 # at least 5% of the image area
    plate_circles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            # Circularity check: actual area vs circle area (at least 50% circularity)
            circular_area = np.pi * (r ** 2)
            if circular_area > 0 and (area / circular_area) > 0.5:
                plate_circles.append((cx, cy, r))

    # Create the full-resolution mask
    mask = np.zeros_like(frame, dtype=np.uint8)

    if plate_circles:
        for cx, cy, r in plate_circles:
            # Scale coordinates back to original resolution
            orig_cx = int(cx / scale)
            orig_cy = int(cy / scale)
            orig_r = int(r / scale)
            
            # Apply edge_size shrinkage
            mask_r = max(0, orig_r - edge_size)
            cv2.circle(mask, (orig_cx, orig_cy), mask_r, 1, -1)
    else:
        # Fallback if no specific circular object is identified
        # Default to a centered circle or full mask if detection fails
        mask.fill(1)

    return mask * 255

def track_and_label_worms(binary_array, persistence=1):
    """
    Tracks and labels worms in a 3D binary array across time.
    
    Args:
        binary_array: 3D Numpy array (time, height, width) of uint8 binary data (0 and 255).
        persistence: Integer for how many frames to copy a worm if it disappears. 
            If a worm doesn't reappear within these frames, the ghosted copies are removed.
            Also serves as the minimum number of frames a label must persist to be retained. 
            Default is 1.

    Returns:
        3D Numpy array (time, height, width) of uint16 labels.
    """
    t_dim, h, w = binary_array.shape
    labeled_array = np.zeros((t_dim, h, w), dtype=np.uint16)
    
    # Track the number of consecutive frames a label has been a "ghost" (copied but not detected)
    ghost_counts = {} # label -> count
    # Track the total number of frames each label appears in for persistence filtering
    label_frame_counts = {} # label -> count
    
    # First frame
    num_labels, labels = cv2.connectedComponents(binary_array[0], connectivity=8)
    labeled_array[0] = labels.astype(np.uint16)
    next_label_id = num_labels
    
    for l in range(1, num_labels):
        ghost_counts[l] = 0
        if persistence > 1:
            label_frame_counts[l] = 1

    # Kernel for dilation in Rule 5
    kernel = np.ones((3, 3), dtype=np.uint8)

    for t in range(1, t_dim):
        prev_labels = labeled_array[t-1]
        curr_binary = binary_array[t]
        
        # Current labeled frame
        curr_frame = np.zeros((h, w), dtype=np.uint16)
        
        # 1. Identify connected components in current binary
        num_curr, labels_curr, stats_curr, _ = cv2.connectedComponentsWithStats(curr_binary, connectivity=8)
        
        # 2. Track which previous labels are "used" by new detections
        used_prev_labels = set()
        
        # 3. Process each component in frame t
        for i in range(1, num_curr):
            mask_i = (labels_curr == i)
            
            # Find unique labels from the previous frame in the current mask's footprint
            overlapping = np.unique(prev_labels[mask_i])
            overlapping = overlapping[overlapping > 0] # Remove background
            
            if len(overlapping) == 0:
                # Rule 3: No overlap, assign new label
                curr_frame[mask_i] = next_label_id
                ghost_counts[next_label_id] = 0
                next_label_id += 1
                if next_label_id == 65535: # Safety check for uint16
                    next_label_id = 1 # We might want to handle this better if it happens
            
            elif len(overlapping) == 1:
                # Rule 2 & 6: Single overlap or split (multiple components overlap same prev label)
                L = overlapping[0]
                curr_frame[mask_i] = L
                used_prev_labels.add(L)
                ghost_counts[L] = 0
            
            else:
                # Rule 5: Multiple overlaps (merge). Partition the current mask.
                # Initialize local partition with seeds from overlapping labels
                partition = np.zeros((h, w), dtype=np.uint16)
                # Seeds are the pixels in mask_i that were labeled in prev_labels
                for L in overlapping:
                    partition[(prev_labels == L) & mask_i] = L
                    used_prev_labels.add(L)
                    ghost_counts[L] = 0
                
                # Expand seeds to fill mask_i
                # We use a simple breadth-first expansion (dilation)
                unfilled_mask_i = mask_i.copy()
                unfilled_mask_i[partition > 0] = False
                
                while unfilled_mask_i.any():
                    # Dilate all labels into the mask
                    dilated = cv2.dilate(partition, kernel)
                    # Find pixels in the mask that were just filled
                    newly_filled = (partition == 0) & (dilated > 0) & mask_i
                    if not newly_filled.any():
                        break
                    partition[newly_filled] = dilated[newly_filled]
                    unfilled_mask_i[newly_filled] = False
                
                curr_frame[mask_i] = partition[mask_i]

        # Rule 4: Handle missing worms (Persistence)
        # Check all labels present in the previous frame
        labels_in_prev = np.unique(prev_labels)
        for L in labels_in_prev:
            if L == 0: continue
            if L not in used_prev_labels:
                # This label disappeared in current binary. 
                # Increment its ghost count and check persistence.
                ghost_counts[L] = ghost_counts.get(L, 0) + 1
                if ghost_counts[L] <= persistence and t < t_dim - 1:
                    # Copy pixels into current frame
                    mask_L = (prev_labels == L)
                    # But only where current frame is still 0 (don't overwrite new detections)
                    curr_frame[mask_L & (curr_frame == 0)] = L
                    # Note: We don't add to used_prev_labels because it didn't find a binary match
                else:
                    # Label's ghost count surpassed persistence. It's officially dead. 
                    # Remove the ghost copies that were previously added to the labeled_array.
                    for t_ghost in range(t - persistence, t):
                        if t_ghost >= 0:
                            labeled_array[t_ghost][labeled_array[t_ghost] == L] = 0
                            # Also decrement its frame count for the final persistence filter
                            if L in label_frame_counts:
                                label_frame_counts[L] -= 1

                    # Label is officially removed from ghost_counts
                    if L in ghost_counts: del ghost_counts[L]
        
        labeled_array[t] = curr_frame
        
        # Track label occurrences in this frame if we need to filter short tracks
        if persistence > 1:
            for L in np.unique(curr_frame):
                if L > 0:
                    label_frame_counts[L] = label_frame_counts.get(L, 0) + 1

    # Final pass to remove short-lived tracks
    if persistence > 1 and label_frame_counts:
        max_label = np.max(labeled_array)
        if max_label > 0:
            # Use a lookup table to efficiently zero out labels with count < persistence
            lookup = np.arange(max_label + 1, dtype=np.uint16)
            for L, count in label_frame_counts.items():
                if count < persistence:
                    if L < len(lookup):
                        lookup[L] = 0
            labeled_array = lookup[labeled_array]

    return labeled_array

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

