import json
import os

def read_crop_coords(subject_id):
    """
    Reads crop coordinates from locator/dataset/bounding_box/{subject_id}_crop_coords.json.
    Returns (start_row, end_row, start_col, end_col) as integers.
    """
    json_path = os.path.join(
        "locator", "dataset", "bounding_box", f"{subject_id}_crop_coords.json"
    )
    with open(json_path, "r") as f:
        coords = json.load(f)
    # row: y_min, y_max; column: x_min, x_max
    start_row = coords["row"]["y_min"]
    end_row = coords["row"]["y_max"]
    start_col = coords["column"]["x_min"]
    end_col = coords["column"]["x_max"]
    return (start_row, end_row, start_col, end_col)

def read_encoding_frequencies(subject_id):
    """
    Reads encoding frequencies (X, Y, Z) from data/preprocessed/raw_json/{subject_id}_dicom_headers.json.
    Returns (X, Y, Z) as floats.
    """
    json_path = os.path.join(
        "data", "preprocessed", "raw_json", f"{subject_id}_dicom_headers.json"
    )
    with open(json_path, "r") as f:
        headers = json.load(f)
    # Take the first available slice key
    first_key = sorted(headers["Encoding Frequency"].keys())[0]
    freq = headers["Encoding Frequency"][first_key]
    return (freq["X"], freq["Y"], freq["Z"])

def read_voxel_sizes(subject_id):
    """
    Reads voxel sizes (x, y, z) from data/preprocessed/raw_json/{subject_id}_dicom_headers.json.
    Returns (x, y, z) as floats.
    """
    json_path = os.path.join(
        "data", "preprocessed", "raw_json", f"{subject_id}_dicom_headers.json"
    )
    with open(json_path, "r") as f:
        headers = json.load(f)
    first_key = sorted(headers["PixelSpacing"].keys())[0]
    pixel_spacing = headers["PixelSpacing"][first_key]  # [x, y]
    slice_thickness = headers["SliceThickness"][first_key]  # z
    return (pixel_spacing[0], pixel_spacing[1], slice_thickness)

# Check consistency of the JSON files (If for a varibale is NOT constant across slices, raise an error)
def check_consistency(subject_id):
    """
    Checks if encoding frequencies and voxel sizes are consistent across all slices.
    Prints a warning if frame numbers are inconsistent.
    Raises ValueError if any inconsistency is found in encoding frequencies or voxel sizes.
    """
    json_path = os.path.join(
        "data", "preprocessed", "raw_json", f"{subject_id}_dicom_headers.json"
    )
    with open(json_path, "r") as f:
        headers = json.load(f)

    # Check encoding frequencies
    enc_freqs = headers["Encoding Frequency"]
    x_set = set()
    y_set = set()
    z_set = set()
    for v in enc_freqs.values():
        x_set.add(v["X"])
        y_set.add(v["Y"])
        z_set.add(v["Z"])
    if len(x_set) > 1 or len(y_set) > 1 or len(z_set) > 1:
        raise ValueError(f"Inconsistent encoding frequencies across slices: X={x_set}, Y={y_set}, Z={z_set}")

    # Check voxel sizes
    pixel_spacings = headers["PixelSpacing"]
    slice_thicknesses = headers["SliceThickness"]
    x_pix_set = set()
    y_pix_set = set()
    z_thick_set = set()
    for k in pixel_spacings:
        x_pix_set.add(pixel_spacings[k][0])
        y_pix_set.add(pixel_spacings[k][1])
        z_thick_set.add(slice_thicknesses[k])
    if len(x_pix_set) > 1 or len(y_pix_set) > 1 or len(z_thick_set) > 1:
        raise ValueError(f"Inconsistent voxel sizes across slices: X={x_pix_set}, Y={y_pix_set}, Z={z_thick_set}")

    # Check frame numbers (warn only)
    frame_numbers = headers["Number of Frames"]
    frame_set = set(frame_numbers[k] for k in frame_numbers)
    if len(frame_set) > 1:
        print(f"⚠️ Warning: Inconsistent frame numbers across slices: {frame_set}")

    print(f"✅ Encoding frequencies and voxel sizes are consistent for {subject_id}.")

def read_slice_locations(subject_id):
    """
    Reads slice locations from data/preprocessed/raw_json/{subject_id}_dicom_headers.json.
    Returns a sorted list of slice locations as floats.
    """
    json_path = os.path.join(
        "data", "preprocessed", "raw_json", f"{subject_id}_dicom_headers.json"
    )
    with open(json_path, "r") as f:
        headers = json.load(f)
    slice_locations = []
    # sort by slice key to ensure correct order
    for key in sorted(headers["Slice Location"].keys()):
        slice_locations.append(headers["Slice Location"][key])
    return slice_locations