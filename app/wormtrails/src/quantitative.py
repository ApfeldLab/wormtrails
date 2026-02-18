import cv2
import numpy as np
from .processing import correct_vignetting, subtract_average

def count_video(video_array, min_size=10, max_size=300, thresh=170, motion_thresh=1, kernel_size=11, detailed_output=False, mask_plate=False):
    """
    Counts the number of living worms in a video array.
    Recordings of 30 seconds to 1 minute are recommended.
    
    Args:
        video_array: 3D Numpy array containing the video frames, with time as axis 0
        min_size: Integer value for the minimum size of a potential worm in pixel area
        max_size: Integer value for the maximum size of a potential worm in pixel area
        thresh: Integer value for the threshold for converting the video array to a binary array
        motion_thresh: Integer value for the threshold for motion detection
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction
        detailed_output: Boolean value on whether to only return the count if False, or to also include visualizations of the detected worms
        mask_plate: Boolean value on whether to mask the plate out from the background
    
    Returns:
        If detailed_output is False:
            An integer value for the number of worms.
        If detailed_output is True:
            A tuple containing the number of worms, the binary array, and the corrected array.
    """
    corrected_array = correct_vignetting(video_array.copy(), kernel_size=kernel_size) # correct spatiotemporal brightness variation
    binary_array = np.where(corrected_array < thresh, 255, 0).astype(np.uint8) # convert to binary, objects darker than surroundings
    if mask_plate: # apply a plate mask to exclude edge artifacts
        binary_array *= create_plate_mask(video_array[0])
    
    motion_array = subtract_average(corrected_array.copy()) # create array containing only pixels which changed in value over the course of the recording

    # Look for objects fitting the size requirements in each frame
    highest_count = 0
    for i in range(video_array.shape[0]): # loop through frames, axis 0 is time in the video array
        filtered_objects, n_objects = filter_objects_by_size(binary_array[i], min_size, max_size, return_count=True)
        if n_objects > highest_count:
            highest_count = n_objects
            highest_count_index = i
            highest_count_objects = filtered_objects

    # Use the presence of motion to validate the objects fitting the size requirements
    validated_objects, n_moving = validate_objects(highest_count_objects, motion_array[highest_count_index], validation_thresh=motion_thresh, return_count=True)

    if detailed_output:
        return n_moving, validated_objects, np.max([corrected_array[highest_count_index], validated_objects], axis=0)
    else:
        return n_moving

def count_frame(frame, reference_frame, min_size=10, max_size=300, thresh=170, motion_thresh=1, kernel_size=11, detailed_output=False, mask_plate=False):
    """
    Counts the number of living worms in a frame.
    
    Args:
        frame: 2D Numpy array containing the frame to be counted
        reference_frame: 2D Numpy array containing the reference frame
        min_size: Integer value for the minimum size of a potential worm in pixel area
        max_size: Integer value for the maximum size of a potential worm in pixel area
        thresh: Integer value for the threshold for converting the frame to a binary array
        motion_thresh: Integer value for the threshold for motion detection
        kernel_size: Odd integer value for the kernel size used to create the blur of the average frame for vignetting correction
        detailed_output: Boolean value on whether to only return the count if False, or to also include visualizations of the detected worms
        mask_plate: Boolean value on whether to mask the plate out from the background
    
    Returns:
        If detailed_output is False:
            An integer value for the number of worms.
        If detailed_output is True:
            A tuple containing the number of worms, the binary array, and the corrected array.
    """
    corrected_frames = correct_vignetting(np.stack([frame, reference_frame], axis=0), kernel_size=kernel_size) # correct spatiotemporal brightness variation
    binary_frame = np.where(corrected_frames[0] < thresh, 255, 0).astype(np.uint8) # convert to binary, objects darker than surroundings
    if mask_plate: # apply a plate mask to exclude edge artifacts
        binary_frame *= create_plate_mask(frame)
    
    motion_frame = subtract_average(corrected_frames.copy())[0] # create array containing only pixels which changed in value over the course of the recording

    # Look for objects fitting the size requirements in the frame
    filtered_objects, n_objects = filter_objects_by_size(binary_frame, min_size, max_size, return_count=True)

    # Use the presence of motion to validate the objects fitting the size requirements
    validated_objects, n_moving = validate_objects(filtered_objects, motion_frame, validation_thresh=motion_thresh, return_count=True)

    if detailed_output:
        return n_moving, validated_objects, np.max([corrected_frames[0], validated_objects], axis=0)
    else:
        return n_moving

def create_plate_mask(frame, edge_size=221):
    """
    Creates a mask for the plate from a raw (uncorrected) still frame.
    Identifies the background as being the brightest part of the image.
    A circular kernel of diameter edge_size is used to expand the background mask to cover the edges of the plate. 
    
    Args:
        frame: 2D Numpy array containing a single raw (uncorrected) video frame.
        edge_size: Odd integer value for the size of the kernel used to create the mask.
    
    Returns:
        A 2D Numpy array containing the mask for the plate, with 1s for the plate and 0s for the background.
    """
    background = np.where(frame > np.max(frame) - 30, 1, 0).astype(np.uint8) # uses bright background to identify plate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_size, edge_size))
    plate_mask = 1 - cv2.dilate(background, kernel)
    return plate_mask

def validate_objects(binary_frame, validation_frame, validation_thresh=1, return_count=False):
    """
    Validates identified objects, given by a binary frame, by requiring that each object surpasses an average threshold value in the validation frame.
    
    Args:
        binary_frame: 2D Numpy array containing the binary video frames, with time as axis 0.
        validation_frame: 2D Numpy array containing the validation video frames, with time as axis 0.
        validation_thresh: Integer value for the threshold for validation.
    
    Returns:
        A binary 2D Numpy array containing only validated objects.
        If return_count is True, returns a tuple of the aformentioned binary array and the count of objects.
    """
    # Find connected components and their statistics
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

    # Count a potential object as a valid moving object if the motion frame shows movement in the object pixels
    valid_objects = np.zeros_like(binary_frame)
    n_valid = 0
    for i in range(1, num_labels):
        mask = (labels == i)
        if validation_frame[mask].mean() > validation_thresh:
            valid_objects[mask] = 255
            n_valid += 1

    if return_count:
        return valid_objects, n_valid
    else:
        return valid_objects

def filter_objects_by_size(binary_frame, min_size, max_size, return_count=False):
    """
    Removes objects from a binary image that do not fall within the specified size range.
    
    Args:
        binary_frame: 2D Numpy array containing the binary video frame. This should be 8-bit unsigned integers, with only 0 and 255 pixel values.
        min_size: Minimum pixel area for an object to be kept.
        max_size: Maximum pixel area for an object to be kept.
        return_count: Boolean value indicating whether to return the count of objects.
    
    Returns:
        A binary 2D Numpy array containing only the objects within the specified size range.
        If return_count is True, returns a tuple of the aformentioned binary array and the count of objects.
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
    filtered_objects = np.where(keep_mask[labels], 255, 0).astype(np.uint8)

    if return_count:
        n_objects = np.sum(keep_mask)
        return filtered_objects, n_objects
    else:
        return filtered_objects
