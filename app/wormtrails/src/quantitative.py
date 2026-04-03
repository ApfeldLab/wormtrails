import cv2
import numpy as np
import pandas as pd
from .processing import correct_vignetting, subtract_average, threshold_array

def count_video(
    video_array, 
    min_size=10, 
    max_size=100, 
    corrected_thresh=203, 
    motion_thresh=3, 
    kernel_size=11,  
    plate_edge_size=None, 
    plate_width=None,
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
        corrected_thresh: Integer value for the threshold for converting the video array to a binary array. If None (default), Otsu's threshold is calculated on masked frames.
        motion_thresh: Integer value for the motion detection threshold. Pixels with motion values above this are considered moving. Default is calculated with Otsu's method.
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction. Default is 11.
        plate_edge_size: Integer value for the width in pixels of the plate edge to be excluded when mask_plate=True. Default is 20% the width of plate_width.
        plate_width: Integer value for the width of the plate in pixels. Default is 80% the width of the frame.
        detailed_output: Boolean value. If False (default), returns only the count. If True, returns a tuple with count, and visualization.
        inPlace: Boolean value. If True, the video array will be modified in place. If False (default), a copy of the video array will be used.

    Returns:
        If detailed_output is False:
            An integer value for the number of living worms detected.
        If detailed_output is True:
            A tuple containing (count, visualization) where:
                - count: Integer count of validated living worms
                - visualization: Image of the plate with living worms highlighted (128) and travelling worms brightly highlighted (255)

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

    motion = np.zeros_like(video_array)
    reference_frame = np.max(video_array, axis=0)
    blur_frame = cv2.medianBlur(reference_frame, kernel_size)
    target_brightness = np.mean(reference_frame)
    for i in range(video_array.shape[0]):
        frame = video_array[i].astype(np.float32)
        frame_brightness = np.mean(frame)
        frame *= (target_brightness / frame_brightness)
        motion[i] = np.abs(frame.copy() - reference_frame).astype(np.uint8)
        video_array[i] = (frame * target_brightness / blur_frame).astype(np.uint8)

    worms = np.zeros_like(video_array)
    worms[video_array < corrected_thresh] = 255
    worms[motion > motion_thresh] = 255
    worms[:, plate_mask == 0] = 0

    worms[worms > 0] = 32
    counts = []
    for t in range(worms.shape[0]):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(worms[t], connectivity=8)

        areas = stats[:, cv2.CC_STAT_AREA]

        moving_labels = labels.copy()
        moving_labels[motion[t] < motion_thresh] = 0
        moving_areas = np.bincount(moving_labels.ravel(), minlength=num_labels)

        is_alive = (areas >= min_size) & (areas <= max_size) & (moving_areas > 0)
        alive_worms = is_alive[labels]
        worms[t, alive_worms] = 128

        is_travelling = (areas >= min_size) & (areas <= max_size) & (moving_areas > areas // 2)
        travelling_worms = is_travelling[labels]
        worms[t, travelling_worms] = 255

        counts.append(np.sum(is_alive.astype(np.uint32)))

    count = int(np.mean(np.array(counts)))

    if detailed_output:
        return count, worms
    else:
        return count

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
        mask.fill()

    return mask * 255

def validate_objects(binary_frame, validation_frame, validation_thresh=1, return_count=False, expand_objects_radius=0):
    """
    Validates identified objects in a binary frame by requiring that each object contains at least one pixel which surpasses a threshold in the validation frame.
    Connected components in the binary frame are tested against the validation frame.
    
    Args:
        binary_frame: 2D Numpy array of 8 bit unsigned integers (uint8) containing the binary frame with potential objects (0 and 255).
        validation_frame: 2D Numpy array of 8 bit unsigned integers (uint8) containing the validation frame to use to decide whether objects are valid.
        validation_thresh: Integer value for the minimum pixel value in the validation frame required to validate an object. Default is 1.
        return_count: Boolean value. If True, returns a tuple of (validated_objects, n_valid). If False, returns only the validated objects.
        expand_objects_radius: Integer value for the radius of the kernel used to expand each object's validation region. Default is 0.

    Returns:
        If return_count is False:
            A binary 2D Numpy array of 8 bit unsigned integers containing only validated objects.
        If return_count is True:
            A tuple of (validated_objects, n_valid) where n_valid is the integer count of validated objects.

    Notes:
        - Each connected component is validated if any pixel in the validation frame exceeds validation_thresh
        - Objects without sufficient motion in the validation frame are removed
    """
    # Find connected components and their statistics
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

    # Count a potential object as a valid moving object if the validation frame surpasses the validation threshold in any of the object pixels
    valid_objects = np.zeros_like(binary_frame)
    n_valid = 0
    for i in range(1, num_labels):
        mask = (labels == i)
        if expand_objects_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_objects_radius*2 + 1, expand_objects_radius*2 + 1))
            mask_expanded = cv2.dilate(mask.astype(np.uint8), kernel)
        else:
            mask_expanded = mask
        if validation_frame[mask_expanded > 0].max() > validation_thresh:
            valid_objects[mask > 0] = 255
            n_valid += 1

    if return_count:
        return valid_objects, n_valid
    else:
        return valid_objects

def filter_objects_by_size(binary_frame, min_size, max_size, return_count=False):
    """
    Removes objects from a binary image that do not fall within the specified size range.
    Connected components are filtered based on their pixel area.
    
    Args:
        binary_frame: 2D Numpy array of 8 bit unsigned integers (uint8) containing the binary frame. Should contain only 0 and 255 pixel values.
        min_size: Minimum pixel area for an object to be kept. Objects smaller than this are removed.
        max_size: Maximum pixel area for an object to be kept. Objects larger than this are removed.
        return_count: Boolean value. If True, returns a tuple of (filtered_objects, n_objects). If False, returns only the filtered objects.

    Returns:
        If return_count is False:
            A binary 2D Numpy array of 8 bit unsigned integers containing only objects within the specified size range.
        If return_count is True:
            A tuple of (filtered_objects, n_objects) where n_objects is the integer count of objects kept.

    Notes:
        - The background (label 0) is always excluded from the size filtering
        - Objects must have area between min_size and max_size (inclusive) to be retained
    """
    # Find connected components and their statistics
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)
    
    # Extract areas for all components (stats[:, 4] is the area column)
    areas = stats[:, cv2.CC_STAT_AREA]
    
    # Create a boolean mask for labels that satisfy the size constraints
    # We explicitly exclude the background (label 0)
    keep_mask = (areas >= min_size) & (areas <= max_size)
    keep_mask[0] = False
    
    # Use the mask to filter the labels and return the binary result
    filtered_objects = threshold_array(keep_mask[labels], 1, dark_objects=False)

    if return_count:
        n_objects = np.sum(keep_mask)
        return filtered_objects, n_objects
    else:
        return filtered_objects

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
