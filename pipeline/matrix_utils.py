import numpy as np
from scipy.sparse import lil_matrix
import numpy.ma as ma

def count_non_zero_rows(input_matrix):
    """
    Utility function to count the number of non-zero rows in a sparse matrix.

    Parameters:
    - input_matrix (scipy.sparse matrix): A sparse matrix in CSR format.

    Returns:
    - int: The number of rows that contain at least one nonzero value.
    """
    non_zero_rows = np.diff(input_matrix.indptr) > 0    # Boolean array where 'True' means the row is not empty
    total_rows = input_matrix.shape[0]                  # Total number of rows in the matrix
    return total_rows - np.sum(~non_zero_rows)          # Count the number of rows that are not empty


def find_closest_grid_points_3d(P0, height, width, depth):
    """
    Find the eight closest grid points around P0 for trilinear interpolation.

    Parameters:
    - P0: The (x, y, z) coordinates of P0.
    - height: Height of the grid.
    - width: Width of the grid.
    - depth: Depth of the grid.

    Returns:
    - list: A list of tuples containing the eight grid points surrounding P0.
    - list: The corresponding interpolation weights.
    """
    x, y, z = P0

    # Identify the eight closest grid points surrounding P0
    x0, x1 = int(np.floor(x)), int(np.floor(x)) + 1
    y0, y1 = int(np.floor(y)), int(np.floor(y)) + 1
    z0, z1 = int(np.floor(z)), int(np.floor(z)) + 1

    grid_points = [(x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)]

    dx, dy, dz = x - x0, y - y0, z - z0
    weights = [(1 - dx) * (1 - dy) * (1 - dz), 
                dx * (1 - dy) * (1 - dz),
                (1 - dx) * dy * (1 - dz),
                dx * dy * (1 - dz),
                (1 - dx) * (1 - dy) * dz,
                dx * (1 - dy) * dz,
                (1 - dx) * dy * dz,
                dx * dy * dz]

    return grid_points, weights


def compute_extreme_p0_z(all_data_unwrapped, subject_id):
    """
    Compute the number of slices needed above and below based on extreme P0 z-values.

    Parameters:
    - all_data_unwrapped (dict): Dictionary with unwrapped phase data.
    - subject_id (str): The subject ID.

    Returns:
    - int: Number of slices needed below.
    - int: Number of slices needed above.
    """
    slices = sorted(all_data_unwrapped[subject_id].keys())
    slice_needed_list = []

    for slice_index, slice_num in enumerate(slices):
        num_frames = all_data_unwrapped[subject_id][slice_num]["phs_z"].shape[2]
        
        min_z_all_frame = np.nanmin(all_data_unwrapped[subject_id][slice_num]["phs_z"][:, :, 0])
        max_z_all_frame = np.nanmax(all_data_unwrapped[subject_id][slice_num]["phs_z"][:, :, 0])

        for frame in range(num_frames):
            min_z_value = np.nanmin(all_data_unwrapped[subject_id][slice_num]["phs_z"][:, :, frame])    # Find the minimum z-value in the frame
            max_z_value = np.nanmax(all_data_unwrapped[subject_id][slice_num]["phs_z"][:, :, frame])    # Find the maximum z-value in the frame

            min_z_all_frame = min(min_z_all_frame, min_z_value)         # Update the minimum z-value across all frames
            max_z_all_frame = max(max_z_all_frame, max_z_value)         # Update the maximum z-value across all frames

        far_above_slice = slice_index + np.ceil(abs(min_z_all_frame))   # as a negative value of z stands for a point above the current slice that this point is coming from
        far_below_slice = slice_index - np.ceil(abs(max_z_all_frame))   # as a positive value of z stands for a point below the current slice that this point is coming from

        slice_needed_list.append([far_above_slice, far_below_slice])    # Based on the current slice index, up to what slice index is needed above and below

    num_above_slice = max([i[0] for i in slice_needed_list]) - (len(slices) - 1)    # substract the last slice index. For example as for the third slice (index=2), we need to go up to the slice index=4. Thus in all we need 4-(3-1)=2 as the last slice index is 3
    num_bottom_slice = abs(min([i[1] for i in slice_needed_list]))                  # no need to substract 1 as the first slice index is 0. At the end as the index are negative, we need to take the absolute value

    return int(num_bottom_slice), int(num_above_slice)


