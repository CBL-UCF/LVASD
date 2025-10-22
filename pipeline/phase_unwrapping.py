import numpy as np
from skimage.restoration import unwrap_phase

def unwrap_phase_data(subject_id, slice_num, all_data):
    """
    Unwrap the phase data (X, Y, Z) for a specific subject and slice using the provided mask,
    ensuring temporal consistency.

    Parameters:
    - subject_id (str): The subject ID.
    - slice_num (int): The slice number.
    - all_data (dict): The nested dictionary containing the loaded data.

    Returns:
    - dict: A dictionary containing the unwrapped phase data for X, Y, and Z.
    """
    data = all_data[subject_id][slice_num]
    phs_x, phs_y, phs_z = data["phs_x"], data["phs_y"], data["phs_z"]
    mask = data["mask"]

    unwrapped_data = {
        "phs_x": np.zeros_like(phs_x),
        "phs_y": np.zeros_like(phs_y),
        "phs_z": np.zeros_like(phs_z)
    }

    def unwrap_roi(wrapped_data_masked):
        """Performs spatial unwrapping in a masked region (single frame)."""
        # As the curent data is wrapped bet -0.5 to 0.5, we need to scale it to -pi to pi
        wrapped_data_scaled = wrapped_data_masked * 2 * np.pi # Scale to [-pi, pi] for the unwrapping function (SciKit-Image)
        unwrapped_scaled = unwrap_phase(wrapped_data_scaled)
        return unwrapped_scaled / (2 * np.pi) # Scale back to [-0.5, 0.5]

    def temporal_unwrap(wrapped_phase_data):
        """Ensures temporal consistency across frames."""
        num_frames = wrapped_phase_data.shape[2]
        unwrapped_phase_data = np.zeros_like(wrapped_phase_data)

        # Unwrap the first frame
        unwrapped_phase_data[:, :, 0] = unwrap_roi(np.ma.array(
            wrapped_phase_data[:, :, 0], mask=(mask[:, :, 0] != 1)
        ))

        for i in range(1, num_frames):
            current_unwrapped = unwrap_roi(np.ma.array(
                wrapped_phase_data[:, :, i], mask=(mask[:, :, i] != 1)
            ))
            delta = current_unwrapped - unwrapped_phase_data[:, :, i - 1] # Compare to the previous frame to see the offset
            valid_pixels = (mask[:, :, i] == 1) 

            if valid_pixels.any(): # If there are valid pixels in the mask
                
                
                computed_offset = np.median(np.asarray(delta[valid_pixels])) 
                rounded_offset = np.round(computed_offset) # Round the offset to the nearest integer
                offset = rounded_offset if abs(rounded_offset) > 0.1 else 0
            else:
                offset = 0

            unwrapped_phase_data[:, :, i] = current_unwrapped - offset # Apply the offset to the current frame

        return unwrapped_phase_data

    # Apply unwrapping for each phase direction
    for key in ["phs_x", "phs_y", "phs_z"]:
        unwrapped_data[key] = temporal_unwrap(data[key]) # Unwrap the phase data

    # Mask invalid regions
    for key in ["phs_x", "phs_y", "phs_z"]:
        unwrapped_data[key][mask != 1] = np.nan # To make sure that the invalid regions are masked

    return unwrapped_data


def unwrap_all_data(all_data): # perform unwrapping for all subjects and slices if needed
    """
    Unwraps all phase data (X, Y, Z) for all subjects and slices.

    Parameters:
    - all_data (dict): The nested dictionary containing the loaded data.

    Returns:
    - dict: A nested dictionary with unwrapped phase data.
    """
    all_data_unwrapped = {}
    for subject_id, slices in all_data.items():
        all_data_unwrapped[subject_id] = {
            slice_num: unwrap_phase_data(subject_id, slice_num, all_data)
            for slice_num in slices
        }
    return all_data_unwrapped


