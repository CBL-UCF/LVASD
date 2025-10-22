import os
import shutil
from glob import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.path import Path
import matplotlib.colors as mcolors


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def save_segmentation_results(subject_id):
    """
    Copies NIfTI outputs and generates visualizations for a subject.
    """
    print(f"📦 Saving results for {subject_id}...")

    # Define colors only for labels (skip background = 0)
    cmap = mcolors.ListedColormap(["green", "blue"])
    bounds = [0.5, 1.5, 2.5]  # 1=myo, 2=LV cavity
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Input paths
    preproc_root = os.path.join("data/preprocessed/raw_nifti", subject_id)
    mask_input_raw = "segmentation/docker_map/dataset/nnUNet_prediction/Dataset071_SCH/labelsTs_ensemble"
    mask_input = "segmentation/docker_map/dataset/nnUNet_prediction/Dataset071_SCH/labelsTs_ensemble_fixed"

    # call the function to fix the mask
    process_nifti_mask(mask_input_raw, mask_input)
    
    # Output folders
    base_out = os.path.join("results", "segmentation")
    nifti_out = os.path.join(base_out, "nifti", subject_id)
    png_out = os.path.join(base_out, "png_masked", subject_id)

    # Create all necessary output folders
    for subfolder in ["Mag", "PhsX", "PhsY", "PhsZ", "Mask"]:
        clear_folder(os.path.join(nifti_out, subfolder))

    # Clear PNG output folder
    clear_folder(png_out)

    # Copy NIfTI files
    for subfolder in ["Mag", "PhsX", "PhsY", "PhsZ"]:
        files = glob(os.path.join(preproc_root, subfolder, "*.nii.gz"))
        for f in files:
            filename = os.path.basename(f)  # Extract the filename
            dst_path = os.path.join(nifti_out, subfolder, filename)  # Full path to the destination file
            shutil.copy2(f, dst_path)  # Copy the file to the destination

    # Sort to ensure consistent order
    mask_files = sorted(glob(os.path.join(mask_input, "*.nii.gz")))
    for f in mask_files:
        filename = os.path.basename(f)
        dst_path = os.path.join(nifti_out, "Mask", filename)
        shutil.copy2(f, dst_path)

    mag_files = sorted(glob(os.path.join(preproc_root, "Mag", "*.nii.gz")))

    for mask_path, mag_path in zip(mask_files, mag_files):
        filename = os.path.basename(mask_path)
        slice_num = int(filename.split("slice")[1].split(".")[0]) # example: subject015_slice08.nii.gz -> 8
        sub_num = int(subject_id.split("_")[1]) # Vol_015 -> 15

        # Load NIfTI files
        mag_nii = nib.load(mag_path).get_fdata()  # shape: (H, W, T)
        mask_nii = nib.load(mask_path).get_fdata()  # shape: (H, W, T)

        # Ensure shapes are compatible
        if mag_nii.shape != mask_nii.shape:
            print(f"❌ Shape mismatch for {subject_id} slice {slice_num}: Mag={mag_nii.shape} vs Mask={mask_nii.shape}")
            continue

        for frame in range(mag_nii.shape[2]):
            mag_frame = mag_nii[:, :, frame]
            mask_frame = mask_nii[:, :, frame]

            # Filter out background values (keep only non-zero values)
            mask_filtered = np.where(mask_frame > 0, mask_frame, np.nan)  # Replace background (0) with NaN

            base_name = f"sub{sub_num:03d}_slc{slice_num:02d}_frm{frame+1:02d}"
            raw_img_path = os.path.join(png_out, f"{base_name}_m.png")
            pred_img_path = os.path.join(png_out, f"{base_name}_pred.png")

            # Overlay: Mag with mask
            plt.figure(figsize=(12, 12), dpi=100)
            plt.imshow(mag_frame, cmap='gray')
            plt.axis('off')
            plt.savefig(raw_img_path, bbox_inches='tight', pad_inches=0)

            # Overlay mask on the Mag image
            plt.imshow(mask_filtered, cmap=cmap, norm=norm, alpha=0.4)
            
            plt.axis('off')
            plt.savefig(pred_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f"✅ Done saving results for {subject_id}")


def process_nifti_mask(input_dir, output_dir):
    """
    Processes NIFTI files in a directory, checks for problematic frames, and fixes them by interpolating or extrapolating boundaries.

    Parameters:
        input_dir (str): Path to the input directory containing NIFTI files.
        output_dir (str): Path to the output directory where modified NIFTI files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # To sort files by name in the input directory
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {file_name}")

            # Load the NIFTI file
            nifti_data = nib.load(file_path)
            data = nifti_data.get_fdata()

            # Process the data to fix problematic frames
            modified_data = check_and_fix_frames(data)

            # Ensure the data type is uint8
            modified_data = modified_data.astype(np.uint8)

            # Save the modified NIFTI file
            output_path = os.path.join(output_dir, file_name)
            modified_nifti = nib.Nifti1Image(modified_data, nifti_data.affine, nifti_data.header)
            nib.save(modified_nifti, output_path)
            print(f"    Saved modified file to: {output_path}")


def check_and_fix_frames(data):
    """
    Checks for problematic frames in a NIFTI file and fixes them by interpolating or extrapolating boundaries.

    Parameters:
        data (numpy.ndarray): The NIFTI data array (height, width, frames).

    Returns:
        numpy.ndarray: The modified data array.
    """
    height, width, frames = data.shape
    modified_data = data.copy()
    valid_frames = []

    # Identify valid frames
    for frame_idx in range(frames):
        mask = data[:, :, frame_idx]
        try:
            # Attempt to extract boundaries
            extract_boundaries(mask)
            valid_frames.append(frame_idx)
        except ValueError:
            pass

    print(f"    Valid frames: {valid_frames}")

    # Handle case where no valid frames exist
    if not valid_frames:
        print("No valid frames found. Skipping file or applying fallback.")
        
        raise ValueError("No valid frames found in the NIFTI file.")

    # Interpolate or extrapolate problematic frames
    for frame_idx in range(frames):
        if frame_idx not in valid_frames:
            print(f"    Fixing frame {frame_idx}...")

            if frame_idx > min(valid_frames) and frame_idx < max(valid_frames):
                # Interpolate using the closest valid frames
                prev_frame_idx = max([f for f in valid_frames if f < frame_idx])
                next_frame_idx = min([f for f in valid_frames if f > frame_idx])
                prev_mask = data[:, :, prev_frame_idx]
                next_mask = data[:, :, next_frame_idx]
                prev_endo, prev_epi = extract_boundaries(prev_mask)
                next_endo, next_epi = extract_boundaries(next_mask)
                weight_prev = (next_frame_idx - frame_idx) / (next_frame_idx - prev_frame_idx)
                weight_next = (frame_idx - prev_frame_idx) / (next_frame_idx - prev_frame_idx)
                endocardium = interpolate_boundaries(prev_endo, next_endo, weight_prev, weight_next)
                epicardium = interpolate_boundaries(prev_epi, next_epi, weight_prev, weight_next)
            elif frame_idx <= min(valid_frames):
                # Extrapolate using the first valid frames
                next_frame_idx = min(valid_frames)
                next_mask = data[:, :, next_frame_idx]
                next_endo, next_epi = extract_boundaries(next_mask)
                endocardium, epicardium = next_endo, next_epi
            elif frame_idx >= max(valid_frames):
                # Extrapolate using the last valid frames
                prev_frame_idx = max(valid_frames)
                prev_mask = data[:, :, prev_frame_idx]
                prev_endo, prev_epi = extract_boundaries(prev_mask)
                endocardium, epicardium = prev_endo, prev_epi

            # Reconstruct the mask for the problematic frame
            modified_data[:, :, frame_idx] = create_mask((height, width), epicardium, endocardium)

    return modified_data


def interpolate_boundaries(boundary1, boundary2, weight1, weight2):
    """
    Interpolates between two sets of boundary points using weights.

    Parameters:
        boundary1 (numpy.ndarray): Boundary points from the first frame.
        boundary2 (numpy.ndarray): Boundary points from the second frame.
        weight1 (float): Weight for the first boundary.
        weight2 (float): Weight for the second boundary.

    Returns:
        numpy.ndarray: Interpolated boundary points.
    """
    return weight1 * boundary1 + weight2 * boundary2


def extract_boundaries(mask):
    """
    Extracts epicardium (LVM) and endocardium (LVC) boundary points from a multi-class mask.

    Parameters:
        mask (numpy.ndarray): The mask array with values:
                              0 = Background, 1 = LVM, 2 = LVC.

    Returns:
        tuple: (endocardium, epicardium) contours as numpy arrays.
    """
    # Separate the mask into two binary masks
    lvm_mask = (mask == 1).astype(np.uint8)  # LVM (epicardium)
    lvc_mask = (mask == 2).astype(np.uint8)  # LVC (endocardium)

    # Find contours for each region
    lvm_contours = find_contours(lvm_mask, level=0.5)
    lvc_contours = find_contours(lvc_mask, level=0.5)

    # Ensure we have at least one contour for each region
    if len(lvm_contours) == 0 or len(lvc_contours) == 0:
        raise ValueError("Could not find both LVM (epicardium) and LVC (endocardium) contours.")

    # Sort contours by size (largest contour first)
    lvm_contours = sorted(lvm_contours, key=lambda c: c.shape[0], reverse=True)
    lvc_contours = sorted(lvc_contours, key=lambda c: c.shape[0], reverse=True)

    # Return the largest contour for each region
    return lvc_contours[0], lvm_contours[0]


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
    mask = np.zeros(shape, dtype=np.float64)
    y_indices, x_indices = np.indices(shape)
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    points = np.vstack((x_indices, y_indices)).T

    path_epi = Path(spline_points_epi)
    path_endo = Path(spline_points_endo)

    mask[path_epi.contains_points(points).reshape(shape)] = 1
    mask[path_endo.contains_points(points).reshape(shape)] = 2

    # Ensure mask values are integers
    mask = np.round(mask).astype(np.uint8)

    return mask


