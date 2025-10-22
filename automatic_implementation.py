
from preprocessing.convert_dicom_to_nifti import convert_subject
from segmentation.utils.organize_nnunet_input import copy_for_nnunet_input
from segmentation.run_nnUNet_prediction import run_nnUNet_prediction
from segmentation.utils.save_segmentation_results import save_segmentation_results
from pipeline.run_strain_computation import run_strain_computation
import sys

def main(subject_id):
    
    print(f"Step 1: Converting DICOM to NIfTI for {subject_id}")
    convert_subject(subject_id)
    
    print(f"Step 2: Copying subject to nnUNet imagesTs")
    copy_for_nnunet_input(subject_id)
    
    print(f"Step 3: Running nnUNet inference")
    run_nnUNet_prediction(subject_id)

    print(f"Step 4: Saving segmentation outputs")
    save_segmentation_results(subject_id)

    print(f"Step 5: Dispalcement & Strain")
    run_strain_computation(subject_id, crop_condition=False, save_unwrap=False)


if __name__ == "__main__":
    subject_id = sys.argv[1]
    main(subject_id)


