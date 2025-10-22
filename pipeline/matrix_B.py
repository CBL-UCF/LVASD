import numpy as np
from scipy.sparse import lil_matrix


def construct_matrix_B(all_data_unwrapped, subject_id, num_botton_slice, num_above_slice):
    """
    Construct matrix B for spatial smoothing in 3D.

    Parameters:
    - all_data_unwrapped: Dictionary containing unwrapped phase data for all subjects and slices.
    - subject_id: The subject ID.

    Returns:
    - Sparse matrix B for spatial smoothing.
    """
    subject_data = all_data_unwrapped[subject_id]
    slices = sorted(subject_data.keys())

    # Initialize variables
    height, width = subject_data[slices[0]]["phs_x"].shape[:2]
    depth = len(slices)  # Original depth
    
    # Compute additional depth based on P0 extremes
    additional_depth = num_botton_slice + num_above_slice
    full_depth = depth + additional_depth
    ngrid = height * width * full_depth

    # Initialize matrix B
    B = lil_matrix((3 * ngrid, ngrid))  # Triple the number of rows for i, j, and k derivatives

    def idx(row, column, k):
        """ Compute flattened index for (row, column, depth) in column-major order """
        if 0 <= row < height and 0 <= column < width and 0 <= k < full_depth:
            return row + column * height + k * height * width
        return None

    # Loop over the grid with margin 1
    margin_grid = 1  # Number of grid points to skip at the edges
    for k in range(margin_grid, full_depth - margin_grid):
        for column in range(margin_grid, width - margin_grid):
            for row in range(margin_grid, height - margin_grid):
                current_idx = idx(row, column, k)
                if current_idx is None:
                    continue

                # Horizontal second derivatives
                left_idx = idx(row, column - 1, k)
                right_idx = idx(row, column + 1, k)
                if left_idx is not None and right_idx is not None:
                    B[current_idx, left_idx] = 1
                    B[current_idx, right_idx] = 1
                    B[current_idx, current_idx] = -2

                # Vertical second derivatives
                up_idx = idx(row - 1, column, k)
                down_idx = idx(row + 1, column, k)
                if up_idx is not None and down_idx is not None:
                    B[current_idx + ngrid, up_idx] = 1
                    B[current_idx + ngrid, down_idx] = 1
                    B[current_idx + ngrid, current_idx] = -2

                # Depth (Z) second derivatives
                back_idx = idx(row, column, k - 1)
                front_idx = idx(row, column, k + 1)
                if front_idx is not None and back_idx is not None:
                    B[current_idx + 2 * ngrid, back_idx] = 1
                    B[current_idx + 2 * ngrid, front_idx] = 1
                    B[current_idx + 2 * ngrid, current_idx] = -2

    return B.tocsr()  # Convert to CSR for efficient computation


