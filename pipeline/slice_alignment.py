from pipeline.compute_resting_frame import compute_resting_frame_using_splines
import numpy as np
from scipy.ndimage import shift
import copy


def register_slices_by_centroid(all_data_unwrapped_voxel_nonRegistered, subject_id):
    """
    Register slices by aligning them based on the centroid of the heart mask.

    Parameters:
        all_data_unwrapped_voxel (dict): Unwrapped Eulerian displacement data.
        subject (str): Subject ID.

    Returns:
        All_data_unwrapped_voxel (dict): Updated unwrapped data with registered slices.
    """
    slice_list = sorted(list(slice_temp_id for slice_temp_id in all_data_unwrapped_voxel_nonRegistered[subject_id].keys()))
    centroids_all = []

    for slice_number in slice_list:
        resting_mask, avg_endo, avg_epi, _, _, PS_frame = compute_resting_frame_using_splines(
            all_data_unwrapped_voxel_nonRegistered, subject_id, slice_number
        )
        resting_mask = resting_mask.astype(int)
        y_coords, x_coords = np.where(resting_mask == 1)
        if len(y_coords) > 0 and len(x_coords) > 0:
            centroid_x = np.mean(x_coords)  # Use float, not int(round())
            centroid_y = np.mean(y_coords)
            print(f"Slice {slice_number}: Myocardium centroid (x, y) = ({centroid_x:.2f}, {centroid_y:.2f})")
            centroids_all.append((centroid_x, centroid_y))
        else:
            print(f"Slice {slice_number}: No myocardium points found.")
            centroids_all.append((None, None))

    height, width = resting_mask.shape
    ref_x = (width - 1) / 2  # True center
    ref_y = (height - 1) / 2

    all_data_unwrapped_voxel_registered = copy.deepcopy(all_data_unwrapped_voxel_nonRegistered)

    for idx, slice_number in enumerate(slice_list):
        centroid = centroids_all[idx]
        if centroid[0] is None:
            continue  # skip slices with no mask

        shift_y = ref_y - centroid[1]
        shift_x = ref_x - centroid[0]

        for key in all_data_unwrapped_voxel_nonRegistered[subject_id][slice_number]:
            arr = all_data_unwrapped_voxel_nonRegistered[subject_id][slice_number][key]
            shifted_arr = np.empty_like(arr)
            for frame in range(arr.shape[2]):
                shifted_arr[..., frame] = shift(
                    arr[..., frame], shift=(shift_y, shift_x), order=0, mode='nearest'
                )
            all_data_unwrapped_voxel_registered[subject_id][slice_number][key] = shifted_arr

    return all_data_unwrapped_voxel_registered