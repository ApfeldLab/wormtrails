import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import Parallel, delayed
from wormtrails.processing import fit_pixel_linear_model

__all__ = [
    'Calibration',
    'count_video',
    'count_simple',
    'create_plate_mask',
    'measure_chemotaxis',
    'measure_chemotaxis_parallel',
    'calculate_relative_metrics',
    'measure_component',
    'measure_window',
]


@dataclass
class Calibration:
    """Calibration for converting pixel/frame units to physical units.

    Attributes:
        pixels_per_mm: Conversion factor — number of pixels per millimeter
            (or per chosen distance unit). Default is 1.0 (no conversion).
        frames_per_second: Frame rate of the recording — number of frames
            per second of real time. Default is 1.0 (no conversion).

    Examples:
        # A typical plate scan with 375 px plate radius and 35 mm plate radius:
        >>> cal = Calibration(pixels_per_mm=375 / 17.5, frames_per_second=1)

        # A 30 fps video:
        >>> cal = Calibration(frames_per_second=30)
    """
    pixels_per_mm: float = 1.0
    frames_per_second: float = 1.0

    def distance_mm(self, pixels: float) -> float:
        """Convert a distance from pixels to millimetres."""
        return pixels / self.pixels_per_mm

    def speed_mm_s(self, pixels_per_frame: float) -> float:
        """Convert a speed from pixels/frame to mm/s."""
        return pixels_per_frame * self.frames_per_second / self.pixels_per_mm

    def area_mm2(self, pixels_area: float) -> float:
        """Convert an area from square pixels to square millimetres."""
        return pixels_area / (self.pixels_per_mm ** 2)

def _find_concentric_center(image, mask_radius):
    """
    Finds the center coordinates of a circular plate feature in the image
    that is of a similar or larger size to the mask_radius.
    
    If no circular feature is found, defaults to the center of the image.

    Args:
        image: 2D Numpy array of uint8.
        mask_radius: Expected radius of the circular feature in pixels.

    Returns:
        cx: Integer x-coordinate of the estimated center.
        cy: Integer y-coordinate of the estimated center.
    """
    h, w = image.shape[:2]
    
    # If the image size is smaller than the mask radius, we cannot expect to find
    # similar or larger circles, so we default to the center of the image.
    if min(h, w) < mask_radius:
        return w // 2, h // 2

    # 1. Try HoughCircles to find similar or larger circles
    try:
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        min_r = int(mask_radius * 0.8)
        max_r = int(mask_radius * 1.1)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            # Pick the first circle detected (typically the most prominent)
            return int(circles[0][0]), int(circles[0][1])
    except Exception:
        pass

    # 2. Fallback: Find contours of thresholded image and fit circles
    try:
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            10
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        best_score = -1
        min_r = mask_radius * 0.8
        max_r = mask_radius * 2.0
        
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            if min_r <= r <= max_r:
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
                score = circularity * r
                if score > best_score:
                    best_score = score
                    best_center = (int(x), int(y))
                    
        if best_center is not None:
            return best_center
    except Exception:
        pass

    # 3. Final fallback: Center of the image
    return w // 2, h // 2

def create_plate_mask(
    image,
    mask_radius=375
):
    """
    Creates a binary plate mask to isolate the assay area from plate boundaries.

    Args:
        image: 2D Numpy array of 8 bit unsigned integers (uint8) of the plate (usually max projection).
        mask_radius: User defined circle radius for the plate mask. Default is 375.

    Returns:
        plate_mask: 2D binary Numpy array (uint8) where 255 represents the assay area and 0 is masked.
    """
    cx, cy = _find_concentric_center(image, mask_radius)
    plate_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(plate_mask, (cx, cy), mask_radius, 255, -1)
    
    return plate_mask

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
    mask_radius=375,
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
        strict_motion_thresh: Higher strict motion threshold. If None, computed as motion_thresh * 1.5. Default is None.
        strict_motion_dilation: Kernel iterations for dilating the strict motion mask. Default is 1.
        stationary_dilation: Kernel iterations for dilating the stationary mask. Default is 1.
        mask_radius: User defined circle radius for the plate mask. Default is 375.
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

    # fit per-pixel linear model across full video; use residuals for motion detection
    residuals, slope, intercept = fit_pixel_linear_model(video)

    # motion projection: mean squared residual per pixel — captures all worm trails
    motion_raw = np.mean(residuals ** 2, axis=0, dtype=np.float32)
    motion_raw[motion_raw < 1] = 1
    log_motion = np.log2(motion_raw.astype(np.float64))
    log_motion *= 5 # improves sensitivity, may need to be adjusted for higher noise recordings
    motion_proj = np.clip(log_motion, 0, 255).astype(np.uint8)

    # per-frame motion: clipped negative residuals — worms darker than linear trend
    # negative residuals indicate the pixel is darker than expected (worm body present now)
    neg_motion = -residuals
    neg_motion[neg_motion < 1] = 1
    neg_motion = neg_motion ** 2
    # log-scale (improves Otsu thresholding)
    neg_log = np.log2(neg_motion.astype(np.float64))
    neg_log *= 5 # match motion_proj scaling
    neg_log = np.clip(neg_log, 0, 255).astype(np.uint8)

    # auto-threshold via Otsu on the motion projection if not specified
    if motion_thresh is None:
        motion_thresh = cv2.threshold(motion_proj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        if motion_thresh < 1:
            motion_thresh = 1
    if strict_motion_thresh is None:
        strict_motion_thresh = min(motion_thresh * 1.5, 255)

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
    plate_mask = create_plate_mask(
        max_proj,
        mask_radius=mask_radius
    )
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
    t_start = video.shape[0] // 3
    t_end = 2 * video.shape[0] // 3
    for t in range(t_start, t_end):
        potential_worms = cv2.adaptiveThreshold(
            video[t].copy(),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            worm_kernel_size,
            worm_thresh
        )
        motion_frame = neg_log[t]

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
            for t in range(t_start, t_end):
                worms_t = worms[t].copy()
                worms_t[labels != l] = 0
                num_labels_t, labels_t, stats_t, centroids_t = cv2.connectedComponentsWithStats(worms_t, connectivity=8)
                areas = stats_t[:, cv2.CC_STAT_AREA]
                is_valid = (areas >= min_worm_area)
                is_valid[0] = False
                label_counts.append(np.sum(is_valid))
                if return_vis and np.max(label_counts) > 0:
                    vis[t, is_valid[labels_t]] = 255
            label_count = np.max(label_counts) if label_counts else 0
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

def count_simple(
    video,
    motion_thresh=1.5,
    dilation_radius=2,
    mask_radius=375,
    return_detail=False,
    calibration=None
):
    """
    Counts worm trails using a simple motion projection threshold.

    Uses linear model residuals to identify motion, thresholds the motion
    projection, dilates to connect nearby pixels, and counts the resulting
    connected components.

    Args:
        video: 3D Numpy array of uint8 with shape (T, H, W).
        motion_thresh: Motion residual threshold for binarisation. Default is 1.5.
        dilation_radius: Radius of the dilation kernel applied to the motion
            mask before counting. Default is 2.
        mask_radius: Radius of the circular plate mask. Default is 375.
        return_detail: If True, returns a DataFrame with per-worm trail distance
            and area. If False, returns just the total count. Default is False.
        calibration: Optional Calibration object for converting pixel units
            to physical units (mm, mm²). Default is None.

    Returns:
        int or pandas.DataFrame:
            If return_detail is False, returns the number of detected worm trails
            (integer).
            If return_detail is True, returns a DataFrame with columns:
            'worm_id', 'distance', 'area' (and 'distance_mm', 'area_mm2' if
            calibration is provided).
    """
    residuals, _, _ = fit_pixel_linear_model(video)
    motion_proj = np.mean(residuals**2, axis=0)
    mask = create_plate_mask(np.mean(video, axis=0).astype(np.uint8), mask_radius=mask_radius)
    motion_proj[mask == 0] = 0
    motion_proj[motion_proj > motion_thresh] = 255
    motion_proj[motion_proj <= motion_thresh] = 0
    motion_proj = motion_proj.astype(np.uint8)
    motion_proj = cv2.dilate(motion_proj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_radius*2+1,dilation_radius*2+1)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_proj)
    # Detailed data per worm
    if return_detail:
        data = []
        for i, (x, y, w, h, area) in enumerate(stats[1:,:]):
            # Minimum enclosing circle
            label_crop = labels[y:y+h, x:x+w]
            pts = np.argwhere(label_crop == (i + 1))
            pts = pts[:, ::-1].astype(np.float32)
            _, radius = cv2.minEnclosingCircle(pts)
            
            entry = {
                'worm_id': i + 1,
                'distance': 2 * (radius - dilation_radius),
                'area': area
            }
            if calibration is not None:
                entry['distance_mm'] = calibration.distance_mm(entry['distance'])
                entry['area_mm2'] = calibration.area_mm2(area)
            data.append(entry)
        df = pd.DataFrame(data)
        return df
    else:
        return num_labels - 1

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

def measure_component(binary_window, component_mask, centroid_yx, time_window, calibration=None):
    """
    Measures movement metrics (position, direction, speed) for a single component in a time window.
    
    Args:
        binary_window: 3D Numpy array of 8 bit unsigned integers (uint8) for the time window, shape (time, height, width).
        component_mask: 2D mask (0 and 255) for the component in the projection.
        centroid_yx: Numpy array of shape (2,) containing (y, x) centroid from the projection.
        time_window: Number of frames in the window used for speed calculation.
        calibration: Optional Calibration object for converting to physical units.
        
    Returns:
        Dictionary with movement metrics for the component, or None if insufficient points to calculate metrics.
        Dictionary keys: 'y', 'x', 'direction_y', 'direction_x', 'speed' (px/frame).
        If calibration is provided, additional keys 'speed_mm_s', 'trail_radius_mm',
        'worm_radius_mm' are included.
        
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

    metrics = {
        'y': position[0],
        'x': position[1],
        'direction_y': direction[0],
        'direction_x': direction[1],
        'speed': speed
    }

    if calibration is not None:
        metrics['speed_mm_s'] = calibration.speed_mm_s(speed)
        metrics['trail_radius_mm'] = calibration.distance_mm(trail_radius)
        metrics['worm_radius_mm'] = calibration.distance_mm(worm_radius)

    return metrics

def measure_window(binary_window, time_window, minimum_size=10, maximum_size=1000, calibration=None):
    """
    Labels connected components in a binary time window and measures movement for each.
    
    Args:
        binary_window: 3D Numpy array of 8 bit unsigned integers (uint8) for the time window, shape (time, height, width).
        time_window: Number of frames in the window used for analysis.
        minimum_size: Minimum pixel area in 2D projection for a component to be considered. Default is 10.
        maximum_size: Maximum pixel area in 2D projection for a component to be considered. Default is 1000.
        calibration: Optional Calibration object for converting to physical units.
        
    Returns:
        List of dictionaries containing movement metrics for each detected component.
        Each dictionary contains: 'y', 'x', 'direction_y', 'direction_x', 'speed', 'label_id', 'time'.
        If calibration is provided, additional keys with physical units are included.
        
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
            metrics = measure_component(binary_window, comp_mask, centroid_yx, time_window, calibration=calibration)
            if metrics:
                metrics['label_id'] = label_id
                window_data.append(metrics)
    return window_data

def measure_chemotaxis(binary_array, time_window=10, interval=60, minimum_size=10, maximum_size=1000, test_spot=None, calibration=None):
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
        calibration: Optional Calibration object for converting pixel/frame units to physical units.
            When provided, additional columns with physical units are added to the output DataFrame.
        
    Returns:
        A pandas DataFrame with one row per detected worm trail, containing columns:
            - 'y', 'x': Position of the component (pixels)
            - 'direction_y', 'direction_x': Direction vector components
            - 'speed': Calculated speed (px/frame)
            - 'time': Starting frame index of the time window
            - 'r': Radial distance to test spot (pixels, if test_spot provided)
            - 'theta': Absolute angle in radians
            - 'relative_angle': Relative angle to test spot in radians
            - 'speed_mm_s': Speed in mm/s (if calibration provided)
            - 'r_mm': Radial distance in mm (if calibration and test_spot provided)
            - 'trail_radius_mm', 'worm_radius_mm': Trail/worm radii in mm (if calibration provided)
            
        Prints progress indicator during processing.

    Notes:
        - Uses sliding window approach with specified overlap
        - Progress is printed to stdout with frame count indicator
    """
    worm_data = []
    
    # Iterate for consecutive time windows
    for t in range(0, binary_array.shape[0] - time_window + 1, interval):
        print(f'{t}/{binary_array.shape[0]}', end="\r")
        
        window_metrics = measure_window(binary_array[t:t+time_window], time_window, minimum_size, maximum_size, calibration=calibration)
        
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
                if calibration is not None:
                    m['r_mm'] = calibration.distance_mm(r)
            
            worm_data.append(m)

    return pd.DataFrame(worm_data)


def measure_chemotaxis_parallel(binary_array, time_window=10, interval=60, minimum_size=10, maximum_size=1000, test_spot=None, calibration=None, n_jobs=-1):
    """Parallel version of measure_chemotaxis using joblib.
    
    Each time window is processed independently, making this ideal for parallelization.
    Falls back to the sequential version when the number of windows is small
    (below the threshold) to avoid parallelization overhead.
    
    Args:
        binary_array: 3D Numpy array of shape (T, H, W) of uint8 (binary 0/255).
        time_window: Number of frames per window (default: 10).
        interval: Frame interval between windows (default: 60).
        minimum_size: Minimum component area in pixels (default: 10).
        maximum_size: Maximum component area in pixels (default: 1000).
        test_spot: Tuple or array of (y, x) for bait location (default: None).
        calibration: Optional Calibration object for converting pixel/frame
            units to physical units.
        n_jobs: Number of parallel workers (-1 for all cores, default: -1).
        
    Returns:
        pandas DataFrame with chemotaxis metrics for each detected worm trail.
        If calibration is provided, additional columns with physical units are included.
    """
    n_windows = (binary_array.shape[0] - time_window + 1) // interval
    
    if n_windows <= 15:
        return measure_chemotaxis(binary_array, time_window, interval, minimum_size, maximum_size, test_spot, calibration)
    
    window_starts = list(range(0, binary_array.shape[0] - time_window + 1, interval))
    
    def _process_window(t):
        window_data = measure_window(binary_array[t:t+time_window], time_window, minimum_size, maximum_size, calibration=calibration)
        for m in window_data:
            m['time'] = t
            if test_spot is not None:
                pos = np.array([m['y'], m['x']])
                direction = np.array([m['direction_y'], m['direction_x']])
                r, theta, rel_angle = calculate_relative_metrics(pos, direction, test_spot)
                m['r'] = r
                m['theta'] = theta
                m['relative_angle'] = rel_angle
                if calibration is not None:
                    m['r_mm'] = calibration.distance_mm(r)
        return window_data
    
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process_window)(t) for t in window_starts
    )
    
    worm_data = []
    for window_results in results:
        worm_data.extend(window_results)
    
    return pd.DataFrame(worm_data)

