import os
import shutil
import numpy as np
import nibabel as nib
import re
import pydicom
from glob import glob
import matplotlib.pyplot as plt
import json

# Utility functions
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


def get_dicom_dirs(dicom_dir, slice_id):
    base_pattern = f"*{slice_id}*"
    dicom_folders = glob(os.path.join(dicom_dir, base_pattern))

    # Filter folders to ensure they belong to the correct slice
    dicom_folders = [folder for folder in dicom_folders if re.search(rf"DENSE{slice_id}", folder)]

    expected_order = ["AveMag", "x-encPha", "y-encPha", "z-encPha"]
    dicom_folders_sorted = []
    for modality in expected_order:
        for folder in dicom_folders:
            if modality in folder:
                dicom_folders_sorted.append(folder)
                break

    return {mod: folder for mod, folder in zip(["mag", "pha_x", "pha_y", "pha_z"], dicom_folders_sorted)}


def load_dicom_images(dicom_folder):
    dicom_files = sorted(glob(os.path.join(dicom_folder, "*.dcm")))
    images = [pydicom.dcmread(f).pixel_array.astype(np.float64) for f in dicom_files]
    return np.stack(images, axis=-1)

def swap_negate_checker(dicom_folder):
    dicom_files = sorted(glob(os.path.join(dicom_folder, "*.dcm")))
    first_image = pydicom.dcmread(dicom_files[0])
    rc_swap, rc_flip = 0, [0, 0, 0]
    if (0x0020, 0x4000) in first_image:
        comment = first_image[0x0020, 0x4000].value
        rc_swap_match = re.search(r"RCswap:(\d+)", comment)
        if rc_swap_match:
            rc_swap = int(rc_swap_match.group(1))
        rc_flip_match = re.search(r"RCSflip:([\d/]+)", comment)
        if rc_flip_match:
            rc_flip = [int(x) for x in rc_flip_match.group(1).split("/")]
    return rc_swap, rc_flip

def images_to_nifti(images, output_path):
    nib.save(nib.Nifti1Image(images, affine=np.eye(4)), output_path)

def save_intermediate_results(mag, phx, phy, phz, output_dir, sub_num, slice_num):
    """
    Save 2x2 plots of modalities (mag, phx, phy, phz) for each frame with colorbars.

    Args:
        mag (numpy.ndarray): Magnitude images (shape: H x W x Frames).
        phx (numpy.ndarray): Phase X images (shape: H x W x Frames).
        phy (numpy.ndarray): Phase Y images (shape: H x W x Frames).
        phz (numpy.ndarray): Phase Z images (shape: H x W x Frames).
        output_dir (str): Directory to save the plots.
        sub_num (int): Subject number.
        slice_num (int): Slice number.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = mag.shape[-1]
    for frame_num in range(num_frames):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plot each modality with colorbars
        im0 = axes[0, 0].imshow(mag[:, :, frame_num], cmap="gray")
        axes[0, 0].set_title("Magnitude")
        axes[0, 0].axis("on")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(phx[:, :, frame_num], cmap="gray")
        axes[0, 1].set_title("Phase X")
        axes[0, 1].axis("on")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(phy[:, :, frame_num], cmap="gray")
        axes[1, 0].set_title("Phase Y")
        axes[1, 0].axis("on")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(phz[:, :, frame_num], cmap="gray")
        axes[1, 1].set_title("Phase Z")
        axes[1, 1].axis("on")
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Save the figure
        output_path = os.path.join(output_dir, f"sub{sub_num:03d}_slc{slice_num:02d}_frm{frame_num+1:02d}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

def frame_number_check(sub_dicom_dir):
    """
    Check if all subdirectories inside the given directory have the same number of DICOM files.

    Args:
        sub_dicom_dir (str): Path to the directory containing subdirectories with DICOM files.

    Returns:
        bool: True if all subdirectories have the same number of DICOM files, False otherwise.
        dict: A dictionary with subdirectory names as keys and their DICOM file counts as values.
    """
    dicom_counts = {}

    # Iterate through all subdirectories
    for subdir in os.listdir(sub_dicom_dir):
        subdir_path = os.path.join(sub_dicom_dir, subdir)
        if os.path.isdir(subdir_path):
            # Count the number of DICOM files in the subdirectory
            dicom_files = glob(os.path.join(subdir_path, "*.dcm"))
            dicom_counts[subdir] = len(dicom_files)

    # Check if all counts are equal
    counts = list(dicom_counts.values())
    all_equal = all(count == counts[0] for count in counts)

    return all_equal, dicom_counts

#  dicom header information
def extract_dicom_header_info(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path)
    header_info = {}
    
    # Voxel size (PixelSpacing and SliceThickness)
    try:
        pixel_spacing = list(ds.PixelSpacing)  # Convert to list for JSON serialization
    except AttributeError:
        pixel_spacing = [None, None]
    slice_thickness = float(getattr(ds, 'SliceThickness', None) or 0)  # Ensure it's a float
    header_info['PixelSpacing'] = pixel_spacing
    header_info['SliceThickness'] = slice_thickness
    
    # ImagePositionPatient and ImageOrientationPatient
    try:
        header_info['ImagePositionPatient'] = list(ds.ImagePositionPatient)
    except AttributeError:
        header_info['ImagePositionPatient'] = None
    try:
        header_info['ImageOrientationPatient'] = list(ds.ImageOrientationPatient)
    except AttributeError:
        header_info['ImageOrientationPatient'] = None


    # Image Size (Rows x Columns)
    header_info['Image Size'] = (
        int(getattr(ds, 'Rows', 0)), 
        int(getattr(ds, 'Columns', 0))
    )
    
    # Slice Location
    header_info['Slice Location'] = float(getattr(ds, 'SliceLocation', None) or 0)
    
    # ImageComments tag (0020,4000) contain Encoding Frequency, Swap XY, Negate flags, and Scale.
    encoding_frequency = None
    swap_xy = None
    negate_flags = None
    scale = None
    if (0x0020, 0x4000) in ds:
        image_comments = ds[(0x0020, 0x4000)].value
        # Encoding Frequency pattern "EncFreq:xxx"
        match = re.search(r"EncFreq:([\d\.]+)", image_comments)
        if match:
            encoding_frequency = float(match.group(1))
            
        # Swap XY pattern "RCswap:x"
        swap_match = re.search(r"RCswap:(\d+)", image_comments)
        if swap_match:
            swap_xy = int(swap_match.group(1))
            
        # Negate flags pattern "RCSflip:x/y/z"
        flip_match = re.search(r"RCSflip:([\d/]+)", image_comments)
        if flip_match:
            negate_flags = [int(x) for x in flip_match.group(1).split('/')]
            
        # Scale pattern "Scale:xxx" (only in phase files)
        scale_match = re.search(r"Scale:([\d\.]+)", image_comments)
        if scale_match:
            scale = float(scale_match.group(1))
        
    header_info['Encoding Frequency'] = encoding_frequency
    header_info['Swap XY'] = swap_xy
    header_info['Negate (X/Y/Z)'] = negate_flags
    header_info['Scale'] = scale

    return header_info

def save_subject_info_to_json(subject_id, output_dir, subject_info):
    """
    Save the cumulative subject information to a JSON file.

    Args:
        subject_id (str): Subject identifier.
        output_dir (str): Directory to save the JSON file.
        subject_info (dict): Dictionary containing the subject's slice information.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{subject_id}_dicom_headers.json")
    with open(output_path, "w") as json_file:
        json.dump(subject_info, json_file, indent=4)
    print(f"✔ Subject information saved to {output_path}")

def process_slice_and_extract_info(slice_id, dicom_folders, subject_info):
    """
    Process a single slice and extract DICOM header information.

    Args:
        slice_id (str): Slice identifier.
        dicom_folders (dict): Dictionary of DICOM folders for the slice.
        subject_info (dict): Dictionary to store cumulative subject information.
    """
    # Extract header info from the magnitude folder
    mag_folder = dicom_folders["mag"]
    first_dicom_file = sorted(glob(os.path.join(mag_folder, "*.dcm")))[0]
    header_info = extract_dicom_header_info(first_dicom_file)

    # Extract Encoding Frequency and Scale from corresponding phase files
    encoding_frequency = {}
    scale = {}
    for direction, folder_key in zip(["X", "Y", "Z"], ["pha_x", "pha_y", "pha_z"]):
        phase_folder = dicom_folders[folder_key]
        first_phase_file = sorted(glob(os.path.join(phase_folder, "*.dcm")))[0]
        phase_header_info = extract_dicom_header_info(first_phase_file)
        encoding_frequency[direction] = phase_header_info["Encoding Frequency"]
        scale[direction] = phase_header_info["Scale"]

    # Calculate the number of frames from the mag
    num_frames = len(sorted(glob(os.path.join(mag_folder, "*.dcm"))))

    # Store the extracted information in the subject info dictionary
    subject_info.setdefault("Number of Frames", {})[slice_id] = num_frames
    subject_info.setdefault("PixelSpacing", {})[slice_id] = header_info["PixelSpacing"]
    subject_info.setdefault("SliceThickness", {})[slice_id] = header_info["SliceThickness"]
    subject_info.setdefault("Image Size", {})[slice_id] = header_info["Image Size"]
    subject_info.setdefault("Slice Location", {})[slice_id] = header_info["Slice Location"]
    subject_info.setdefault("Encoding Frequency", {})[slice_id] = encoding_frequency
    subject_info.setdefault("Swap XY", {})[slice_id] = header_info["Swap XY"]
    subject_info.setdefault("Negate (X/Y/Z)", {})[slice_id] = header_info["Negate (X/Y/Z)"]
    subject_info.setdefault("Scale", {})[slice_id] = scale
    subject_info.setdefault("ImagePositionPatient", {})[slice_id] = header_info["ImagePositionPatient"]
    subject_info.setdefault("ImageOrientationPatient", {})[slice_id] = header_info["ImageOrientationPatient"]


#  Main conversion function
def convert_subject(subject_id, input_root="data/raw_dicom", output_root="data/preprocessed/raw_nifti"):
    sub_dicom_dir = os.path.join(input_root, subject_id)
    sub_output_dir = os.path.join(output_root, subject_id)

    # Check frame number consistency across subdirectories
    print()
    all_equal, dicom_counts = frame_number_check(sub_dicom_dir)
    if all_equal:
        print("✔ All subdirectories have the same number of DICOM files.")
    else:
        print("✘ Subdirectories have different numbers of DICOM files:")
        for subdir, count in dicom_counts.items():
            print(f"  - {subdir}: {count} files")
    print()

    # Create subfolders of output NIfTI folders
    mag_dir_nifti = os.path.join(sub_output_dir, "Mag")
    phx_dir_nifti = os.path.join(sub_output_dir, "PhsX")
    phy_dir_nifti = os.path.join(sub_output_dir, "PhsY")
    phz_dir_nifti = os.path.join(sub_output_dir, "PhsZ")

    # Clear the output folders if they exist
    for folder in [mag_dir_nifti, phx_dir_nifti, phy_dir_nifti, phz_dir_nifti]:
        clear_folder(folder)

    # Initialize a dictionary to store subject DICOM header information
    subject_info = {}

    # Extract slice IDs and group folders
    slice_dict = {}
    for folder_name in os.listdir(sub_dicom_dir):
        match = re.search(r"DENSE(\d{2})", folder_name)
        if match:
            slice_id = match.group(1)  # Extract the two digits after "DENSE"
            if slice_id not in slice_dict:
                slice_dict[slice_id] = []
            slice_dict[slice_id].append(folder_name)
    
    # Sort slice IDs
    slice_dict = {k: v for k, v in sorted(slice_dict.items(), key=lambda item: int(item[0]))}

    # Sort folders by expected order (mag, pha_x, pha_y, pha_z)
    for slice_id, folders in slice_dict.items():
        folders.sort(key=lambda x: ("AveMag" in x, "x-encPha" in x, "y-encPha" in x, "z-encPha" in x))
        slice_dict[slice_id] = folders
    
    # Images folder for saving intermediate results
    sub_images_dir = sub_output_dir.replace("raw_nifti", "raw_images")
    clear_folder(sub_images_dir)

    # Process each slice
    for slice_id in slice_dict:
        folders = slice_dict[slice_id]
        if len(folders) < 4:
            print(f"Skipping slice {slice_id}: missing folders.")
            continue

        # Get the DICOM folders for this slice
        dicom_folders = get_dicom_dirs(sub_dicom_dir, slice_id)

        # Extract and store DICOM header information
        process_slice_and_extract_info(slice_id, dicom_folders, subject_info)

        # Get the swap and negate information from the mag folder
        rc_swap, rc_negate = swap_negate_checker(dicom_folders["mag"])

        # Load the DICOM images and normalize them between [0, 1] (as the original images are 12-bit)
        mag = load_dicom_images(dicom_folders["mag"]) / 4095.0
        if rc_swap == 0:
            phx = load_dicom_images(dicom_folders["pha_x"]) / 4095.0
            phy = load_dicom_images(dicom_folders["pha_y"]) / 4095.0
            phz = load_dicom_images(dicom_folders["pha_z"]) / 4095.0
        else:
            phx = load_dicom_images(dicom_folders["pha_y"]) / 4095.0
            phy = load_dicom_images(dicom_folders["pha_x"]) / 4095.0
            phz = load_dicom_images(dicom_folders["pha_z"]) / 4095.0

        # Bring the images to the range to [-0.5, 0.5]
        phx -= 0.5
        phy -= 0.5
        phz -= 0.5

        # Negate the phase images if necessary
        if rc_negate[0]: phx = -phx
        if rc_negate[1]: phy = -phy
        if rc_negate[2]: phz = -phz

        # Get the subject number and slice number from the subject_id
        slice_num = int(slice_id)
        sub_num = int(subject_id.split("_")[1])

        # Save intermediate results as 2x2 plots
        save_intermediate_results(mag, phx, phy, phz, sub_images_dir, sub_num, slice_num)

        # Save the images as NIfTI files
        images_to_nifti(mag, os.path.join(mag_dir_nifti, f"subject{sub_num:03d}_slice{slice_num:02d}_0000.nii.gz"))
        images_to_nifti(phx, os.path.join(phx_dir_nifti, f"subject{sub_num:03d}_slice{slice_num:02d}_0001.nii.gz"))
        images_to_nifti(phy, os.path.join(phy_dir_nifti, f"subject{sub_num:03d}_slice{slice_num:02d}_0002.nii.gz"))
        images_to_nifti(phz, os.path.join(phz_dir_nifti, f"subject{sub_num:03d}_slice{slice_num:02d}_0003.nii.gz"))

        print(f"✔ Processed {subject_id} slice {slice_num}")
    
    # Save the cumulative subject information to a JSON file
    raw_json_dir = output_root.replace("raw_nifti", "raw_json")
    save_subject_info_to_json(subject_id, raw_json_dir, subject_info)
