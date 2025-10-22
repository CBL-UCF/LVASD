import numpy as np
from scipy.sparse import lil_matrix
from pipeline.matrix_utils import find_closest_grid_points_3d

def construct_matrix_A(all_data_unwrapped, subject_id, frame_number, num_bottom_slice, num_above_slice):
    """
    Construct matrix A using unwrapped phase data from a specific frame for a specific subject.

    Parameters:
    - all_data_unwrapped: Dictionary containing unwrapped phase data for all subjects and slices.
    - subject_id: The subject ID.
    - frame_number: The frame number to process.

    Returns:
    - Sparse matrix A where each row corresponds to a non-NaN point in the unwrapped data.
    """
    subject_data = all_data_unwrapped[subject_id]
    slices = sorted(subject_data.keys())
    
    # Get spatial dimensions
    height, width = subject_data[slices[0]]["phs_x"].shape[:2]
    depth = len(slices) # Number of slices (excluding padding)
    full_depth = depth + num_bottom_slice + num_above_slice # Total depth including padding
    ngrid = height * width * full_depth # Total number of grid points
    
    # Track valid indices for real values
    real_values_indices = np.zeros((height, width, full_depth), dtype=bool) # In full depth
    
    # Mark valid indices
    for slice_index, slice_num in enumerate(slices):
        z_position = slice_index + num_bottom_slice # Adjusted for padding
        Xunwrap_frame = subject_data[slice_num]["phs_x"][:, :, frame_number]
        real_values_indices[:, :, z_position] = ~np.isnan(Xunwrap_frame) # Mark non-NaN values as True in full depth
    
    n = np.sum(real_values_indices)
    A = lil_matrix((n, ngrid)) # Initialize sparse matrix A
    row_index = 0
    
    for slice_index, slice_num in enumerate(slices):
        z_position = slice_index + num_bottom_slice # Adjusted for padding
        Xunwrap_frame = subject_data[slice_num]["phs_x"][:, :, frame_number]
        Yunwrap_frame = subject_data[slice_num]["phs_y"][:, :, frame_number]
        Zunwrap_frame = subject_data[slice_num]["phs_z"][:, :, frame_number]
        
        for col in range(width): # column changes across the width
            for row in range(height): # row changes across the height
                if real_values_indices[row, col, z_position]:
                    initial_matrix = np.zeros((height, width, full_depth)) # Initialize of ngird size that at the end will be flattened and useed as a single row in A
                    
                    # Compute P0
                    P0_x = col - Xunwrap_frame[row, col] # Use negative behind $$ As the right-hand is the positive direction in matrix indexing of Python (considering the positive value of the phase being in the left of the myocardium in the unwrapped phase)
                    P0_y = row + Yunwrap_frame[row, col] # Use positive behind $$ As the down direction is the positive direction in matrix indexing of Python (considering the negative value of the phase being in the top of the myocardium in the unwrapped phase)
                    P0_z = z_position - Zunwrap_frame[row, col] # Use negative behind $$ As the negative value of the phase stands for a myocardium point that is coming from the top of the grid
                    P0 = (P0_x, P0_y, P0_z)
                    
                    # Get 8 closest grid points
                    grid_points, weights = find_closest_grid_points_3d(P0, height, width, full_depth)
                    
                    for (x, y, z), weight in zip(grid_points, weights):
                        if 0 <= y < height and 0 <= x < width and 0 <= z < full_depth:
                            initial_matrix[y, x, z] = weight  # Swap x and y as the location of y stands for row and x stands for column in the matrix
                    
                    flattened_matrix = initial_matrix.flatten(order='F') # Flatten the matrix in column-major order
                    non_zero_indices = flattened_matrix.nonzero()[0] # Get the indices of non-zero elements
                    A[row_index, non_zero_indices] = flattened_matrix[non_zero_indices] # as A is n*ngrid, so we need to fill the row_index row of A with the non-zero values of the flattened_matrix
                    row_index += 1 # Move to the next row in A
    
    return A.tocsr()  # Convert to CSR for efficiency