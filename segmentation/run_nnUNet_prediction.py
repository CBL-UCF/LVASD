import subprocess
import os


def run_nnUNet_prediction(subject_id):
    """
    Runs nnUNetv2_predict in Docker for the given subject and performs ensembling.
    Assumes all environment variables and volumes are mounted correctly.
    """
    project_root = os.path.abspath(".")
    docker_map_path = os.path.join(project_root, "segmentation", "docker_map")

    image_name = "nn_unet"
    image_tag = "0.0.1"

    # Paths for predictions from each fold
    fold_prediction_paths = []
    for fold in range(5):  # 5 folds (0 to 4)
        fold_output_path = f"/workspace/docker_map/dataset/nnUNet_prediction/Dataset071_SCH/labelsTs_fold{fold}"
        fold_prediction_paths.append(fold_output_path)

        docker_cmd = [
            "docker", "run", "--rm", "--ipc=host", "--gpus", "all",
            "-v", f"{docker_map_path}:/workspace/docker_map",
            "-e", "nnUNet_results=/workspace/docker_map/dataset/nnUNet_results",
            "-e", "nnUNet_preprocessed=/workspace/docker_map/dataset/nnUNet_preprocessed",
            "-e", "nnUNet_raw=/workspace/ignore",  # just a dummy path to make it shut up
            f"{image_name}:{image_tag}",
            "nnUNetv2_predict",
            "-d", "Dataset071_SCH",
            "-i", "/workspace/docker_map/dataset/nnUNet_raw/Dataset071_SCH/imagesTs",
            "-o", fold_output_path,
            "-f", str(fold),
            "-c", "3d_fullres",
            "--save_probabilities"
        ]

        print(f"🚀 Running nnUNetv2_predict for fold {fold} via Docker...")
        subprocess.run(docker_cmd, check=True)
        print(f"✅ nnUNet inference complete for fold {fold}.")

    # Perform ensembling
    ensemble_output_path = "/workspace/docker_map/dataset/nnUNet_prediction/Dataset071_SCH/labelsTs_ensemble"
    ensemble_cmd = [
        "docker", "run", "--rm", "--ipc=host", "--gpus", "all",
        "-v", f"{docker_map_path}:/workspace/docker_map",
        f"{image_name}:{image_tag}",
        "nnUNetv2_ensemble",
        "-i", *fold_prediction_paths,  # Use -i for input directories
        "-o", ensemble_output_path
    ]

    print("🚀 Running nnUNetv2_ensemble to combine predictions...")
    subprocess.run(ensemble_cmd, check=True)
    print("✅ nnUNet ensembling complete.")