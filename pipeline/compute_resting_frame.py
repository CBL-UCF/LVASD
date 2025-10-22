import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy.spatial import cKDTree
from matplotlib.path import Path
import scipy.ndimage as ndimage

def extract_boundaries(mask):
    """Extracts epicardium and endocardium boundary points from a binary mask."""
    contours = find_contours(mask, level=0.5)  # Find all contours

    if len(contours) < 2:
        raise ValueError("Could not find both epicardium and endocardium contours.")

    # Sort contours based on standard deviation along Y-axis to differentiate endo & epi
    contours = sorted(contours, key=lambda c: np.std(c[:, 0]), reverse=True)
    endocardium, epicardium = contours[::-1]  # Ensure first is endo, second is epi

    return endocardium, epicardium


def fit_spline(boundary_points, num_points=100):
    """Fits a smooth spline curve to boundary points."""
    x, y = boundary_points[:, 1], boundary_points[:, 0]  # Convert to (x, y) so swaped
    tck, u = interp.splprep([x, y], s=5, per=True)  # Fit spline, per = True for periodic boundary to close loop smoothly
    x_smooth, y_smooth = interp.splev(np.linspace(0, 1, num_points), tck)
    return np.vstack([y_smooth, x_smooth]).T


def get_myocardium_points(mask):
    """Extracts valid myocardium points (where we have non-NaN displacements)."""
    myo_points = np.column_stack(np.where(mask))  # Get (row, col) indices
    return myo_points


def find_closest_myocardium(boundary_points, myocardium_points):
    """Finds the closest myocardium point for each boundary point."""
    tree = cKDTree(myocardium_points)  # Build KD-Tree for fast lookup
    distances, indices = tree.query(boundary_points)  # Find nearest myo points
    closest_myo_points = myocardium_points[indices]  # Get closest myo coordinates
    return closest_myo_points, distances  # Return matched points + distances


def process_heart_mask(all_data_unwrapped_voxel, subject, slice_number, frame_number):
    """Finds myocardium points closest to epicardium & endocardium."""
    mask = ~np.isnan(all_data_unwrapped_voxel[subject][slice_number]['phs_x'][:, :, frame_number])
    
    # Step 1: Extract epicardium & endocardium
    endocardium, epicardium = extract_boundaries(mask)

    # Step 2: Extract myocardium points
    myocardium_points = get_myocardium_points(mask)

    # Step 3: Find closest myocardium points for endocardium & epicardium
    endo_myo, endo_dists = find_closest_myocardium(endocardium, myocardium_points)
    epi_myo, epi_dists = find_closest_myocardium(epicardium, myocardium_points)

    return endo_myo, epi_myo, endo_dists, epi_dists


def compute_resting_frame_using_splines(all_data_unwrapped_voxel, subject, slice_number):
    """
    Computes the initial heart mask by averaging the original epicardium & endocardium across all frames.

    Parameters:
        all_data_unwrapped_voxel (dict): Unwrapped Eulerian displacement data.
        subject (str): Subject ID.
        slice_number (int): Slice Number (Not Necessarily Equal To Slice Index).

    Returns:
        numpy.ndarray: The final mask containing the initial heart region.
    """
    slice_list = list(slice_temp_id for slice_temp_id in all_data_unwrapped_voxel[subject].keys())

    slices_peak_systole = []
    for slice_temp_id in slice_list:
        peak_frame, _ = peak_systole_finder(all_data_unwrapped_voxel, subject, slice_temp_id)
        slices_peak_systole.append(peak_frame)

    # Find the median peak systole frame across all slices
    # We also need to exclude 0 as it appears when there is no predicted segmentation mask for a specific frame of a slice
    peak_frame_median = np.median(slices_peak_systole).astype(int) # Median of peak systole frames across slices as the common frame
    print(f"Common peak systole frame (median): {peak_frame_median + 1}/ {all_data_unwrapped_voxel[subject][slice_number]['phs_x'].shape[2]}")

    original_endo_list = []
    original_epi_list = []

    all_frames = all_data_unwrapped_voxel[subject][slice_number]['phs_x'].shape[2]

    for frame_number in range(all_frames): # Loop over all frames to compute splines (Still, we will average up to peak systole frame)

        print(f"Processing frame {frame_number + 1}/{all_frames} for subject {subject}, slice {slice_number}")

        try: 
            # Find closest myocardium points
            endo_myo, epi_myo, _, _ = process_heart_mask(all_data_unwrapped_voxel, subject, slice_number, frame_number)
            
            # Extract endo & epi boundaries
            mask = ~np.isnan(all_data_unwrapped_voxel[subject][slice_number]['phs_x'][:, :, frame_number])
            endocardium, epicardium = extract_boundaries(mask)

            # Get X and Y Eulerian displacement fields
            Xunwrap = all_data_unwrapped_voxel[subject][slice_number]['phs_x'][:, :, frame_number]
            Yunwrap = all_data_unwrapped_voxel[subject][slice_number]['phs_y'][:, :, frame_number]

            # Compute original locations for endocardium
            original_endo = []
            for endo, myo in zip(endocardium, endo_myo):
                myo_x, myo_y = int(myo[1]), int(myo[0]) # Convert to int as indices are integers
                if 0 <= myo_x < Xunwrap.shape[1] and 0 <= myo_y < Xunwrap.shape[0]:  # Ensure valid index
                    P0_x = endo[1] - Xunwrap[myo_y, myo_x]
                    P0_y = endo[0] + Yunwrap[myo_y, myo_x]
                    original_endo.append([P0_y, P0_x])
            original_endo = np.array(original_endo)

            # Compute original locations for epicardium
            original_epi = []
            for epi, myo in zip(epicardium, epi_myo):
                myo_x, myo_y = int(myo[1]), int(myo[0])
                if 0 <= myo_x < Xunwrap.shape[1] and 0 <= myo_y < Xunwrap.shape[0]:  # Ensure valid index
                    P0_x = epi[1] - Xunwrap[myo_y, myo_x]
                    P0_y = epi[0] + Yunwrap[myo_y, myo_x]
                    original_epi.append([P0_y, P0_x])
            original_epi = np.array(original_epi)

            # Fit splines
            endo_spline = fit_spline(original_endo)
            epi_spline = fit_spline(original_epi)

            # Store results
            original_endo_list.append(endo_spline)
            original_epi_list.append(epi_spline)
        
        except ValueError as e: # Handle case where contours cannot be found
            print(f"    Slice {slice_number}, Frame {frame_number + 1}/{all_frames} is skipped: {e}")
            continue  # Skip frames where contours cannot be found

    # Compute the average splines up to peak systole frame
    avg_endo = np.mean(np.array(original_endo_list[0:peak_frame + 1]), axis=0) # +1 to include peak systole frame as well
    avg_epi = np.mean(np.array(original_epi_list[0:peak_frame + 1]), axis=0) # +1 to include peak systole frame as well


    # Generate final mask
    mask_shape = all_data_unwrapped_voxel[subject][slice_number]['phs_x'].shape[:2]
    final_mask = create_mask(mask_shape, avg_epi, avg_endo)

    return final_mask, avg_endo, avg_epi, original_endo_list, original_epi_list, peak_frame_median


def create_mask(shape, spline_points_epi, spline_points_endo):
    """
    Create a binary mask with epicardium as 1 and endocardium as 2.

    Parameters:
        shape (tuple): Shape of the mask (height, width).
        spline_points_epi (ndarray): Interpolated epicardium points.
        spline_points_endo (ndarray): Interpolated endocardium points.

    Returns:
        numpy.ndarray: Mask with values 1 (epicardium) and 2 (endocardium).
    """
    # Initialize the mask
    mask = np.zeros(shape, dtype=np.float64)

    # Ensure indices are aligned to pixel centers
    y_indices, x_indices = np.indices(shape)  # Grid indices
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    points = np.vstack((x_indices, y_indices)).T  # Combine coordinates

    #  Swap x and y for Path compatibility
    spline_points_epi = np.column_stack((spline_points_epi[:, 1], spline_points_epi[:, 0]))  # Swap x and y
    spline_points_endo = np.column_stack((spline_points_endo[:, 1], spline_points_endo[:, 0]))  # Swap x and y

    # Define paths for epicardium and endocardium
    path_epi = Path(spline_points_epi)
    path_endo = Path(spline_points_endo)

    # Fill mask based on paths
    mask[path_epi.contains_points(points).reshape(shape)] = 1  # Epicardium
    mask[path_endo.contains_points(points).reshape(shape)] = 2  # Endocardium

    return mask


def peak_systole_finder(all_data_unwrapped_voxel, subject, slice_number):
    """
    Finds the frame corresponding to peak systole by identifying the frame 
    with the smallest cavity size inside the myocardium ring.

    Parameters:
    - all_data_unwrapped_voxel (dict): Dictionary with unwrapped Eulerian data.
    - subject (str): Subject ID.
    - slice_number (int): Slice Number.

    Returns:
    - int: Frame index corresponding to peak systole.
    - list: Cavity sizes for each frame.
    """
    num_frames = all_data_unwrapped_voxel[subject][slice_number]['phs_x'].shape[2]
    cavity_sizes = []  # Store cavity voxel counts per frame

    for frame_number in range(num_frames):
        # Create myocardium mask (True = myocardium, False = cavity or background)
        mask = ~np.isnan(all_data_unwrapped_voxel[subject][slice_number]['phs_x'][:, :, frame_number])

        # Fill the inside cavity to separate it from the background
        filled_mask = ndimage.binary_fill_holes(mask)  # Now True = myocardium + cavity

        # Extract the cavity: filled region minus the myocardium
        cavity_region = filled_mask & ~mask  # True for cavity, False elsewhere

        # Count the number of cavity voxels
        cavity_size = np.count_nonzero(cavity_region)

        # Store cavity size for this frame
        cavity_sizes.append(cavity_size)

    # Find the frame with the smallest cavity size (peak systole) 
    # Exlude frames with cavity size 0 (no mask predicted)
    valid_indices = [i for i, size in enumerate(cavity_sizes) if size > 0]
    if valid_indices:
        valid_cavity_sizes = [cavity_sizes[i] for i in valid_indices]
        peak_systole_frame = valid_indices[np.argmin(valid_cavity_sizes)]
        print(f"Slice {slice_number}: Peak systole occurs at frame {peak_systole_frame+1}/{num_frames} with cavity size {cavity_sizes[peak_systole_frame]}")
    else:
        peak_systole_frame = 0  # Default to first frame if none are valid
        print(f"Slice {slice_number}: No valid cavity found, defaulting to frame 1.")
    
    return peak_systole_frame, cavity_sizes


