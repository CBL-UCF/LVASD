import os
import shutil
from glob import glob

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

def copy_for_nnunet_input(subject_id):
    """
    Copy all .nii.gz files from Mag, PhsX, PhsY, PhsZ into the nnUNet imagesTs folder.
    """
    print(f"Copying NIfTI files for {subject_id} to nnUNet imagesTs...")

    input_root = os.path.join("data/preprocessed/raw_nifti", subject_id)
    # check if the input directory exists
    if not os.path.exists(input_root):
        print(f"❌ Input directory {input_root} does not exist. Skipping {subject_id}.")
        return

    # Clear the output folder before copying
    output_folder_images = "segmentation/docker_map/dataset/nnUNet_raw/Dataset071_SCH/imagesTs"
    clear_folder(output_folder_images)

    # Comulate all folds path that exist in segmentation/docker_map/dataset/nnUNet_prediction/Dataset055_SCH
    output_folder_labels = "segmentation/docker_map/dataset/nnUNet_prediction/Dataset071_SCH"
    for item in os.listdir(output_folder_labels):
        item_path = os.path.join(output_folder_labels, item)
        if os.path.isdir(item_path):
            clear_folder(item_path)

    # Copy files from each modality folder as the raw input for nnUNet
    for modality in ["Mag", "PhsX", "PhsY", "PhsZ"]:
        modality_path = os.path.join(input_root, modality)
        nii_files = sorted(glob(os.path.join(modality_path, "*.nii.gz")))

        for file_path in nii_files:
            filename = os.path.basename(file_path)
            dst_path = os.path.join(output_folder_images, filename)
            shutil.copy2(file_path, dst_path)

    print(f"✅ Done copying files for {subject_id}")
