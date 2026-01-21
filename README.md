# Left Ventricle Automated Strain from DENSE (LVASD) MRI

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-required-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Overview 🔭

This repository provides a fully automated, end-to-end computational pipeline for the analysis of cardiac DENSE (Displacement Encoding with Stimulated Echoes) MRI. The workflow processes raw DICOM series and executes all necessary steps—including preprocessing, deep learning-based segmentation, phase unwrapping, and tensor calculations—to yield final displacement and strain maps.

The pipeline is designed for batch processing and requires minimal user intervention beyond initial data placement.

---

## 2. Core Components 🧩

* **Preprocessing Module**: Automates the conversion of multi-slice, multi-phase DICOM files into a structured NIfTI format suitable for analysis.
* **LV Segmentation**: Integrates a pre-trained **nnU-Net** model to perform robust, automated segmentation of the left ventricle across all cardiac phases.
* **Displacement & Strain Engine**: Implements algorithms for phase unwrapping, displacement field calculation, and Lagrangian strain tensor computation.
* **Structured Output**: All results, including NIfTI files, segmentation masks, and final displacement & strain evaluations, are stored in a logically organized directory structure for traceability and further analysis.

---

## 3. System Requirements 💻

* **Operating System**: Linux (recommended) or any OS capable of running Docker.
* **Docker Engine**: Must be installed and running. GPU support (via NVIDIA Container Toolkit) is highly recommended for performance.
* **Python**: Version 3.8+ (required only to run the asset download script).
* **Git**: For cloning the repository.

---

## 4. Installation and Setup (One-Time) 🛠️

Follow these steps to prepare the environment.

### Step 4.1: Clone the Repository 

    ```bash
    git clone <https://github.com/CBL-UCF/LVASD>
    cd LVASD
    ```
### Step 4.2: Install Docker

If you do not have Docker installed, follow the official instructions for your OS:
* **[Install Docker Engine](https://docs.docker.com/engine/install/)**

For GPU acceleration, you must also install the NVIDIA Container Toolkit:
* **[NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**

### Step 4.3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4.4: Download Model Weights and Docker Image

Run the provided Python script. This will download the nnU-Net model weights (`docker_map` folder) and the pre-built Docker image (`nn_unet_image.tar`).

```bash
python download_docker.py
```

### Step 4.5: Load the Docker Image

Load the downloaded image into your local Docker registry.

```bash
docker load -i nn_unet_image.tar
```

To verify, run `docker images`. You should see `nn_unet:0.0.1` in the list. The setup is now complete.

---

## 5. Execution Protocol 🚀

### Step 5.1: Data Staging

Place your subject's DICOM data into the `data/raw_dicom/` directory, following the strict hierarchical structure below:

```
data/raw_dicom/
└── Vol_***/                          # e.g., Vol_012           : Voluntee 012 
    ├─────────── DENSE**_AveMag/      # e.g., DENSE04_AveMag    : Slice 04 magnitude dicom files
    ├─────────── DENSE**_x-encPha/    # e.g., DENSE04_x-encPha  : Slice 04 phase X dicom files
    ├─────────── DENSE**_y-encPha/    # e.g., DENSE04_y-encPha  : Slice 04 phase Y dicom files
    ├─────────── DENSE**_z-encPha/    # e.g., DENSE04_z-encPha  : Slice 04 phase Z dicom files
    └── ...                           # e.g.,                   : Other Slices with structure simialr to above
```

### Step 5.2: Run the Full Pipeline 

Execute the main script from the project's root directory. Provide the subject's folder name as a command-line argument.

```bash
python automatic_implementation.py <subject_id>
```

**Example:**
```bash
python automatic_implementation.py Vol_012
```

The script will handle everything: preprocessing the data, preparing it for nnU-Net, **automatically calling Docker to perform the segmentation**, running the strain analysis, and saving the results.

### Step 5.3: Accessing Results

All final outputs are saved to your local `results/` directory.

---

## 6. Codebase Architecture 📂

```
LVASD/
├── automatic_implementation.py       # Main pipeline controller - RUN THIS
├── download_docker.py                # Setup script for assets
├── nn_unet_image.tar                 # Docker image (after download)
├── requirements.txt
├── data/
│   ├── raw_dicom/                    # INPUT: Raw DICOM data
│   └── processed/                    # Processed INPUT
│   │   ├── raw_images/               # All frames PNG files (magnitude + 3 phases)
│   │   ├── raw_json/                 # Subjects' dicom header info
│   │   └── raw_nifti/                # Data stored in NIFTI format to pass to segmenation sectio
├── segmentation/
│   ├── run_nnUNet_prediction.py      # (Called automatically) Wrapper for Docker
│   └── docker_map/                   # nnU-Net model weights (after download)
├── pipeline/
└── results/                          # OUTPUT: Final analysis results 
│   ├── displacement/                 # Displacement at original voxel centroid (vtk & npy)
│   ├── displacement_query/           # Displacement at query points (vtk & npy)
│   ├── phase_unwrapping/             # Phase Unwrapped vs. Phase Wrapped frames (png)  -- set save_unwrap=True in the automatic implementation
│   ├── resting_mask/                 # Resting Mask for each slice
│   ├── segmentation/                 # Segmentaiton Results
│   └── strain/                       # Strains at query points (point-wise strain, region-wise strain, slice-wise strain, slice-wise median, and strin maps at peak systole)

```

---

## 7. Citation 📖

*(                                 )*

## 8. License 🔑

This project is licensed under the MIT License. See the `LICENSE` file for details.
