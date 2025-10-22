import os
import numpy as np
import nibabel as nib

def load_all_data(base_dir):
    """
    Load magnitude, phase (X, Y, Z), and mask data from structured subfolders.

    Parameters:
    - base_dir (str): Root directory containing subject-specific subfolders.

    Returns:
    - all_data (dict): Nested dictionary storing data by subject and slice.
                       Format: {subject_id: {slice_num: {'mag': {ndarray}, 'mask': ndarray, 'phs_x': ndarray, 'phs_y': ndarray, 'phs_z': ndarray}}}
    
    * ndarray: [height, width, frames]
    * the data is flipped vertically to match physical coordinate system
    """
    # Define subfolder names
    subfolders = {
        "mag": "Mag",
        "mask": "Mask",
        "phs_x": "PhsX",
        "phs_y": "PhsY",
        "phs_z": "PhsZ"
    }

    # Initialize a dictionary to hold all the data
    all_data = {}

    # Helper function to extract subject ID and slice number from filename
    def extract_ids(filename): # e.g. subject017_slice05_0001.nii.gz
        parts = filename.split('_') # split by '_'
        subject_id = parts[0] # subject017
        slice_num = int(parts[1][5:7]) # slice05
        return subject_id, slice_num

    # Iterate through each subfolder
    for data_type, subfolder in subfolders.items():
        folder_path = os.path.join(base_dir, subfolder)
        files = sorted(os.listdir(folder_path))
        for filename in files:
            if filename.endswith(".nii.gz"):
                subject_id, slice_num = extract_ids(filename)

                # Initialize subject dictionary if not already present
                if subject_id not in all_data:
                    all_data[subject_id] = {}

                # Initialize slice dictionary if not already present
                if slice_num not in all_data[subject_id]:
                    all_data[subject_id][slice_num] = {}

                # Load the data
                file_path = os.path.join(folder_path, filename)
                data = nib.load(file_path).get_fdata()
                data = np.flipud(data)  # Flip the data vertically to match physical coordinate system
                all_data[subject_id][slice_num][data_type] = data

    return all_data

