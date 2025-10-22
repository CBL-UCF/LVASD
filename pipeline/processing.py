import numpy as np
import copy


def replace_nan_with_zero(data_dict):
    """
    Recursively replaces all NaN values in the given dictionary with 0.
    
    Parameters:
    - data_dict (dict): A dictionary that may contain NaN values.
    
    Returns:
    - updated_dict (dict): A dictionary with all NaN values replaced by 0.
    """
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            data_dict[key] = replace_nan_with_zero(value)
    elif isinstance(data_dict, np.ndarray):
        # Replace NaN values in numpy arrays
        data_dict = np.nan_to_num(data_dict, nan=0.0)
    
    return data_dict


def get_crop_coords_from_masks(all_data, buffer):
    """
    Find the overall bounding box (start_row, end_row, start_col, end_col)
    covering all mask pixels with value 1.0 across all subjects, slices, and frames.
    """

    # Get the height and width from any mask
    height, width, _ = next(iter(next(iter(all_data.values())).values()))['mag'].shape
    buffer_height = int(buffer * height)
    buffer_width = int(buffer * width)
    
    min_row, min_col = np.inf, np.inf
    max_row, max_col = -np.inf, -np.inf

    for subject_id, slices in all_data.items():
        for slice_num, data in slices.items():
            mask = data['mask']  # shape: (H, W, F)
            mask_indices = np.where(mask == 1.0)
            if mask_indices[0].size > 0:
                min_row = min(min_row, np.min(mask_indices[0]))
                max_row = max(max_row, np.max(mask_indices[0]))
                min_col = min(min_col, np.min(mask_indices[1]))
                max_col = max(max_col, np.max(mask_indices[1]))

    if min_row == np.inf:  # No mask found
        raise ValueError("No mask pixels with value 1.0 found in the dataset.")

    # Add buffer and ensure coordinates are within image bounds
    return int(min_row) - buffer_height, int(max_row) + buffer_height, int(min_col) - buffer_width, int(max_col) + buffer_width


def crop_data(all_data, crop_buffer, crop_condition=True):
    """
    Crop the magnitude, phase (X, Y, Z), and mask data for all subjects and slices.

    Parameters:
    - all_data (dict): A nested dictionary containing the loaded data.
                       Format: {subject_id: {slice_num: data_dict}}
    - crop_coords (tuple): Coordinates to crop the data in the format (start_row, end_row, start_col, end_col).

    Returns:
    - cropped_data (dict): A nested dictionary containing the cropped data.
                           Format: {subject_id: {slice_num: data_dict}}
    """
    
    start_row, end_row, start_col, end_col = get_crop_coords_from_masks(all_data, crop_buffer)

    cropped_data = {}

    for subject_id, slices in all_data.items():
        cropped_data[subject_id] = {}
        for slice_num, data in slices.items():
            cropped_data[subject_id][slice_num] = {}
            for data_type, array in data.items(): # data_type: 'mag', 'mask', 'phs_x', 'phs_y', 'phs_z'
                
                # Perform cropping if crop_condition is True
                if crop_condition:
                    cropped_data[subject_id][slice_num][data_type] = array[start_row:end_row, start_col:end_col, :]
                else:
                    cropped_data[subject_id][slice_num][data_type] = array

    updated_cropped_data = replace_nan_with_zero(cropped_data) # Replace NaN values with 0, otherwise the unwrapped function 
    #                                                            will fail (trapped in an infinite loop and never return)

    return updated_cropped_data


def multiply_non_nan_values(data_dict, factors):
    """
    Multiply non-NaN values in the 'phs_x', 'phs_y', and 'phs_z' arrays by given factors.

    Parameters:
    - data_dict (dict): Dictionary containing the data.
                        Format: {subject_id: {slice_num: {'phs_x': ndarray, 'phs_y': ndarray, 'phs_z': ndarray}}}
    - factors (tuple): Tuple containing the multiplication factors for 'phs_x', 'phs_y', and 'phs_z'.

    Returns:
    - new_data_dict (dict): Updated dictionary with multiplied values.
    """
    factor_x, factor_y, factor_z = factors

    # Create a deep copy of the original dictionary
    new_data_dict = copy.deepcopy(data_dict)

    for subject_id, slices in new_data_dict.items():
        for slice_num, data in slices.items():
            if 'phs_x' in data:
                data['phs_x'][~np.isnan(data['phs_x'])] *= factor_x
            if 'phs_y' in data:
                data['phs_y'][~np.isnan(data['phs_y'])] *= factor_y
            if 'phs_z' in data:
                data['phs_z'][~np.isnan(data['phs_z'])] *= factor_z

    return new_data_dict


def compute_sparsity(matrix):
    """
    Computes the sparsity of a given matrix.

    Parameters:
    - matrix (numpy.ndarray or scipy.sparse matrix): The input matrix.

    Returns:
    - float: Sparsity ratio (0 to 1), where 1 means fully sparse.
    """
    height, width = matrix.shape
    total_elements = height * width
    non_zero_elements = matrix.nnz if hasattr(matrix, 'nnz') else np.count_nonzero(matrix)
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity