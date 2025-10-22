
import os
import tarfile
import gdown

# Function to download and extract .tar files
def download_and_extract_tar(url, extract_to, filename="downloaded_file.tar", file_untar=True):
    """
    Downloads a .tar file from the given URL and extracts it to the specified folder.

    Args:
        url (str): The URL of the .tar file to download.
        extract_to (str): The folder where the .tar file should be extracted.
        filename (str): The name to save the downloaded file as.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(extract_to, exist_ok=True)

        # Define the local path for the downloaded file
        tar_file_path = os.path.join(extract_to, filename)

        # Download the .tar file
        print(f"Downloading {url}...")
        gdown.download(url, tar_file_path, quiet=False)
        print(f"Downloaded to {tar_file_path}")

        if file_untar:
            # Extract the .tar file
            print(f"Extracting {tar_file_path}...")
            with tarfile.open(tar_file_path, "r") as tar:
                tar.extractall(path=extract_to)
            print(f"Extracted to {extract_to}")

            # Optionally, delete the .tar file after extraction
            os.remove(tar_file_path)
            print(f"Removed the tar file: {tar_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



# Use the function to download and extract the required folders
if __name__ == "__main__":

    # Importing docker_map folder
    tar_url = "https://drive.google.com/uc?id=1yS5CqZ6n8lO1xAbGBydPps9arPsDFIfR"  # Direct download link
    destination_folder = "segmentation" # To the segmentation folder
    download_and_extract_tar(tar_url, destination_folder, filename="docker_map.tar", file_untar=True)

    # Importing nnUNet_image folder
    nnUNet_image_url = "https://drive.google.com/uc?id=1qM5HuJYAU0pMUL4UFGHY3jjUFIodgHg0" # Direct download link
    destination_folder = "." # To the main directory
    download_and_extract_tar(nnUNet_image_url, destination_folder, filename="nn_unet_image.tar", file_untar=False)


