import cv2
import numpy as np
import pandas as pd
from wormtrails.processing import fit_pixel_linear_model

def count_video(
    video_array,
    min_worm_area=20,
    max_worm_area=300,
    max_worm_length=30,
    worm_kernel_size=11,
    worm_thresh=5,
    motion_thresh=None,
    strict_motion_thresh=None,
    strict_motion_dilation=1,
    stationary_dilation=1,
    edge_contrast_kernel_size=51,
    edge_contrast_thresh=10,
    mask_inclusion_kernel_size=31,
    edge_offset=3,
    return_vis=True
):
    """
    Counts roaming and stationary living worms in a video using linear model residuals and connected components analysis.
    
    Args:
        video_array: 3D Numpy array of 8 bit unsigned integers (uint8) containing the video frames, with time as axis 0.
        min_worm_area: Minimum pixel area in 2D projection to count a worm. Default is 20.
        max_worm_area: Maximum pixel area in 2D projection to count a worm. Default is 300.
        max_worm_length: Maximum expected worm length in pixels, used for dilation of problematic worm motion. Default is 30.
        worm_kernel_size: Kernel size for adaptive thresholding to detect worm bodies. Default is 11.
        worm_thresh: Threshold offset for adaptive thresholding. Default is 5.
        motion_thresh: Motion threshold for the log-scaled motion projection. If None, computed via Otsu from the log-motion distribution. Default is None.
        strict_motion_thresh: Higher strict motion threshold. If None, computed as motion_thresh + 0.5 (after Otsu). Default is None.
        strict_motion_dilation: Kernel iterations for dilating the strict motion mask. Default is 1.
        stationary_dilation: Kernel iterations for dilating the stationary mask. Default is 1.
        edge_contrast_kernel_size: Kernel size for edge contrast detection (plate mask). Default is 51.
        edge_contrast_thresh: Threshold for edge contrast detection. Default is 10.
        mask_inclusion_kernel_size: Kernel size for plate mask morphological closing. Default is 31.
        edge_offset: Number of erosion iterations on the plate mask edge. Default is 3.
        return_vis: If True, returns a visualization array with detected worms highlighted. If False, returns the original video copy. Default is True.

    Returns:
        n_roaming: Integer count of roaming (moving) worms detected across all frames.
        n_stationary_alive: Integer count of stationary but alive worms detected.
        vis: Visualization array with detected worms highlighted (255 for roaming, 128 for stationary).

    Notes:
        Motion is detected using per-pixel linear model residuals (from fit_pixel_linear_model)
        instead of raw frame differencing. Motion energy is log-scaled for dynamic range compression.
        Per-frame motion uses clipped negative residuals to localize worms at their current position.
        Thresholds are automatically set via Otsu on the log-motion distribution when set to None.
        Prints per-label progress to stdout during processing.
    """
    # currently fixed parameters:
    small_kernel_size = 3
    worm_dilation_kernel_size = 5

    # load video
    video = video_array.copy()

    # create kernels which we'll use throughout the pipeline
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_kernel_size, small_kernel_size))
    worm_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (worm_dilation_kernel_size, worm_dilation_kernel_size))
    mask_inclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_inclusion_kernel_size, mask_inclusion_kernel_size))

    # fit per-pixel linear model across full video; use residuals for motion detection
    residuals, slope, intercept = fit_pixel_linear_model(video)

    # motion projection: mean squared residual per pixel — captures all worm trails
    motion_raw = np.mean(residuals ** 2, axis=0)
    motion_raw[motion_raw < 1] = 1
    log_motion = np.log2(motion_raw.astype(np.float64))
    if np.max(log_motion) > 0:
        log_motion = log_motion * 255.0 / np.max(log_motion)
    motion_proj = np.clip(log_motion, 0, 255).astype(np.uint8)

    # per-frame motion: clipped negative residuals — worms darker than linear trend
    # negative residuals indicate the pixel is darker than expected (worm body present now)
    neg_motion = -residuals
    neg_motion[neg_motion < 0] = 0
    # log-scale each frame independently
    neg_log = np.zeros_like(neg_motion, dtype=np.uint8)
    for t in range(neg_motion.shape[0]):
        ft = neg_motion[t]
        ft[ft < 1] = 1
        lt = np.log2(ft.astype(np.float64))
        if np.max(lt) > 0 and motion_thresh is None:
            lt = lt * 255.0 / np.max(lt)
        neg_log[t] = np.clip(lt, 0, 255).astype(np.uint8)

    # auto-threshold via Otsu on the motion projection if not specified
    if motion_thresh is None:
        motion_thresh = cv2.threshold(motion_proj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        if motion_thresh < 2:
            motion_thresh = 2
    if strict_motion_thresh is None:
        strict_motion_thresh = min(motion_thresh + 10, 255)

    max_proj = np.max(video, axis=0)
    median_proj = np.median(video, axis=0).astype(np.uint8)

    problematic_pixels = max_proj.copy()
    problematic_pixels = cv2.adaptiveThreshold(
        problematic_pixels,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        worm_kernel_size,
        worm_thresh
    )
    problematic_pixels[motion_proj <= motion_thresh] = 0

    # get stationary objects
    stationary = median_proj.copy()
    stationary = cv2.adaptiveThreshold(
        stationary,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        worm_kernel_size,
        worm_thresh
    )

    # create mask to remove edges
    plate_mask = max_proj.copy()
    seed_point = [plate_mask.shape[0]//2, plate_mask.shape[1]//2]
    plate_mask = cv2.adaptiveThreshold(
        plate_mask,
        128,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        edge_contrast_kernel_size,
        edge_contrast_thresh
    )
    cv2.circle(plate_mask, seed_point, 100, 128, -1)
    cv2.floodFill(plate_mask, None, seed_point, 255, 0, 0, flags=4)
    plate_mask[plate_mask < 255] = 0
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, mask_inclusion_kernel)
    plate_mask = cv2.erode(plate_mask, small_kernel, iterations=edge_offset)
    motion_proj[plate_mask == 0] = 0
    stationary[plate_mask == 0] = 0

    # threshold
    ret, strict_motion_mask = cv2.threshold(motion_proj.copy(), strict_motion_thresh, 255, cv2.THRESH_BINARY)
    ret, motion_mask = cv2.threshold(motion_proj.copy(), motion_thresh, 255, cv2.THRESH_BINARY)

    # remove small noise and expand mask for roaming worm detection
    for noise_size_thresh in [1, 15]:
        _, mask_labels, mask_stats, _ = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
        areas = mask_stats[:, cv2.CC_STAT_AREA]
        areas[0] = 0
        motion_mask[(areas <= noise_size_thresh)[mask_labels]] = 0
        motion_mask = cv2.dilate(motion_mask, small_kernel)
    motion_mask = cv2.erode(motion_mask, small_kernel, iterations=2)
    stationary[motion_mask > 0] = 255
    motion_mask[plate_mask == 0] = 0
    strict_motion_mask[plate_mask == 0] = 0

    # find living worms which will be used for counting the number of worms in trails
    worms = np.zeros_like(video, dtype=np.uint8)
    for t in range(worms.shape[0]):
        potential_worms = cv2.adaptiveThreshold(
            video[t].copy(),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            worm_kernel_size,
            worm_thresh
        )
        motion_frame = neg_log[t].astype(np.float64)

        num_labels_t, labels_t, stats_t, centroids_t = cv2.connectedComponentsWithStats(potential_worms, connectivity=8)
        moving_labels_t = np.unique(labels_t[motion_frame > motion_thresh])
        is_moving = np.isin(np.arange(num_labels_t), moving_labels_t)
        is_fully_moving = ~np.isin(np.arange(num_labels_t), np.unique(labels_t[motion_frame <= motion_thresh]))
        problematic_labels_t = np.unique(labels_t[problematic_pixels > 0])
        is_problematic = np.isin(np.arange(num_labels_t), problematic_labels_t)
        areas = stats_t[:, cv2.CC_STAT_AREA]
        # first, find worms which are above the minimum area and moving
        is_valid = (areas >= min_worm_area) & is_moving & (areas <= max_worm_area) & (~is_problematic | is_fully_moving)
        is_valid[0] = False
        worms[t, is_valid[labels_t]] = 255

        # recalculate motion using median-based negative residuals for fallback
        median_frame = np.abs(median_proj.copy().astype(np.int16) - video[t].copy().astype(np.int16)).astype(np.float64)
        is_moving_median = np.isin(np.arange(num_labels_t), np.unique(labels_t[median_frame > motion_thresh]))
        # of the valid worms, check whether they are problematic and not fully moving
        is_fallback = (areas >= min_worm_area) & is_moving_median & is_problematic & ~is_fully_moving
        is_fallback[0] = False
        # if they are problematic and not fully moving, add their associated motion pixels to a separate frame
        fallback_motion = (median_frame > strict_motion_thresh) & is_fallback[labels_t]
        fallback_motion = fallback_motion.astype(np.uint8)
        # dilate the problematic/large worm motion pixels by half the maximum length of a worm
        for i in range(int(max_worm_length/(worm_dilation_kernel_size-1))):
            fallback_motion = cv2.dilate(fallback_motion, worm_dilation_kernel)
            fallback_motion[motion_mask == 0] = 0
        worms[t, fallback_motion > 0] = 255
    worms[:, plate_mask == 0] = 0

    # loop through trails and find the number of worms in each one
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
    n_roaming = 0
    if return_vis:
        vis = video.copy()
        vis[:, motion_proj > motion_thresh] = 0
    for l in range(1, num_labels):
        trail_area = stats[l, cv2.CC_STAT_AREA]
        if trail_area > max_worm_area/2 or (trail_area > min_worm_area and np.max(problematic_pixels[labels==l]) == 0):
            label_counts = []
            for t in range(video.shape[0]):
                worms_t = worms[t].copy()
                worms_t[labels != l] = 0
                num_labels_t, labels_t, stats_t, centroids_t = cv2.connectedComponentsWithStats(worms_t, connectivity=8)
                areas = stats_t[:, cv2.CC_STAT_AREA]
                is_valid = (areas >= min_worm_area)
                is_valid[0] = False
                label_counts.append(np.sum(is_valid))
                if return_vis and np.max(label_counts) > 0:
                    vis[t, is_valid[labels_t]] = 255
            label_count = np.max(label_counts)
            n_roaming += label_count
            if label_count > 0:
                strict_motion_mask[labels == l] = 0
                stationary[labels == l] = 0
                print(f"Label {l}: {label_count}  ", end="\r")
                if return_vis:
                    vis[(vis != 255) & (labels == l)] = 128
    print(" "*60, end="\r")

    # find stationary alive worms from the strict motion mask
    alive_stationary = np.zeros_like(stationary, dtype=np.uint8)
    num_labels_sw, labels_sw, stats_sw, _ = cv2.connectedComponentsWithStats(stationary, connectivity=8)

    overlapping_labels = np.unique(labels_sw[cv2.dilate(strict_motion_mask, small_kernel, iterations=strict_motion_dilation) > 0])
    for label_idx in overlapping_labels:
        area = stats_sw[label_idx, cv2.CC_STAT_AREA]
        if label_idx == 0:
            continue
        elif area >= min_worm_area and area <= max_worm_area:
            alive_stationary[labels_sw == label_idx] = 255
        elif area > max_worm_area:
            alive_stationary[(motion_mask > 0) & (labels_sw == label_idx)] = 255
    alive_stationary = cv2.dilate(alive_stationary, small_kernel, iterations=stationary_dilation)

    n_quiescent, _, _, _ = cv2.connectedComponentsWithStats(alive_stationary, connectivity=8)
    n_quiescent -= 1

    if return_vis:
        vis[(vis != 255) & (alive_stationary > 0)] = 255
        video = vis

    return n_roaming, n_quiescent, video

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

