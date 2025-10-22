import numpy as np
import scipy.sparse.linalg as spla
import copy
import time
from scipy.sparse import identity, vstack
from pipeline.io_utils import load_all_data
from pipeline.processing import crop_data, multiply_non_nan_values, compute_sparsity
from pipeline.phase_unwrapping import unwrap_all_data
from pipeline.phase_unwrapping_exporter import save_phase_unwrap_intermediates
from pipeline.matrix_A import construct_matrix_A
from pipeline.matrix_B import construct_matrix_B
from pipeline.matrix_utils import count_non_zero_rows, compute_extreme_p0_z
from pipeline.slice_alignment import register_slices_by_centroid


def compute_lagrangian_displacement(base_directory, subject_id, encoding_frequencies, voxel_sizes, lambda_=3, mu=0.05, crop_buffer=0.08, crop_condition=True, save_unwrap = True):
    """
    Computes Lagrangian displacement from phase data.

    Returns:
        lagrangian_displacements_voxel (height, width, full_depth, num_frames, 3): Lagrangian displacements in voxel units.
        all_data_unwrapped_voxel (height, width, full_depth, num_frames, 3): Unwrapped phase data in voxel units.
        lagrangian_displacements_mm (height, width, full_depth, num_frames, 3): Lagrangian displacements in mm.
        all_data_unwrapped_mm (height, width, full_depth, num_frames, 3): Unwrapped phase data in mm.
        num_bottom_slice (int): Number of slices below the first slice.
        num_above_slice (int): Number of slices above the last slice.
    """
    print("\nStarting Lagrangian Displacement Computation...")
    start_time_all = time.time()

    # Load data
    all_data = load_all_data(base_directory)
    all_data_cropped = crop_data(all_data, crop_buffer=crop_buffer, crop_condition=crop_condition)
    all_data_unwrapped = unwrap_all_data(all_data_cropped)

    if save_unwrap:
        # Save intermediate unwrapping results
        print("Saving phase unwrapping intermediates...")
        save_phase_unwrap_intermediates(all_data_cropped, all_data_unwrapped, subject_id)
        print("Phase unwrapping intermediates saved.")


    # Define encoding frequencies (Hz) and voxel sizes (mm)
    encoding_frequency_x, encoding_frequency_y, encoding_frequency_z = encoding_frequencies
    voxel_size_x, voxel_size_y, voxel_size_z = voxel_sizes

    # Compute voxel scaling factors
    factor_x = 1 / (encoding_frequency_x * voxel_size_x)
    factor_y = 1 / (encoding_frequency_y * voxel_size_y)
    factor_z = 1 / (encoding_frequency_z * voxel_size_z)
    factors_to_voxel = (factor_x, factor_y, factor_z)

    
    # Scale unwrapped data to voxel units and register slices
    all_data_unwrapped_voxel_nonRegistered = multiply_non_nan_values(all_data_unwrapped, factors_to_voxel) 
    all_data_unwrapped_voxel = register_slices_by_centroid(all_data_unwrapped_voxel_nonRegistered, subject_id) # Register slices by centroid

    # Also prepare mm version
    all_data_unwrapped_mm = copy.deepcopy(all_data_unwrapped_voxel)
    factors_to_mm = (voxel_size_x, voxel_size_y, voxel_size_z)
    all_data_unwrapped_mm = multiply_non_nan_values(all_data_unwrapped_mm, factors_to_mm)

    # Extract spatial dimensions and time frames
    slices = sorted(all_data_unwrapped_voxel[subject_id].keys())

    # Assume all slices have the same height and width
    height, width = all_data_unwrapped_voxel[subject_id][slices[0]]["phs_x"].shape[:2]
    num_frames = min(all_data_unwrapped_voxel[subject_id][s]["phs_x"].shape[2] for s in slices) # Ensure that the number of frames is consistent across all slices
    depth = len(slices)

    # Compute padding in the Z direction
    num_bottom_slice, num_above_slice = compute_extreme_p0_z(all_data_unwrapped_voxel, subject_id) # Compute padding in the Z direction
    full_depth = depth + num_bottom_slice + num_above_slice

    # Initialize Lagrangian displacement array
    lagrangian_displacements_voxel = np.zeros((height, width, full_depth, num_frames, 3))

    # Compute matrix B once
    B_matrix = construct_matrix_B(all_data_unwrapped_voxel, subject_id, num_bottom_slice, num_above_slice)
    
    # Start the loop for the frames
    A_matrices = {}

    for frame_number in range(num_frames):
        
        # Draft start time
        start_time = time.time()

        # Construct Matrix A
        if frame_number not in A_matrices:
            A_matrix = construct_matrix_A(all_data_unwrapped_voxel, subject_id, frame_number, num_bottom_slice, num_above_slice)
            A_matrices[frame_number] = A_matrix
        else:
            A_matrix = A_matrices[frame_number]

        # Construct identity matrix as sparse
        identity_matrix = identity(A_matrix.shape[1], format='csr')

        # Compute alpha for scaling B
        alpha = np.sqrt(A_matrix.shape[0] / count_non_zero_rows(B_matrix))

        # modify the B for different directions coeeficients
        ngrid = B_matrix.shape[1]  # Since B_matrix is (3ngrid, ngrid)
        B_x = B_matrix[:ngrid, :]
        B_y = B_matrix[ngrid:2 * ngrid, :]
        B_z = B_matrix[2 * ngrid:, :]
        lambda_xy = lambda_  # Same lambda for X and Y
        lambda_z = lambda_ * 0.5 if lambda_ > 2 else 1.1 # Different lambda for Z
        B_x_scaled = lambda_xy * alpha * B_x
        B_y_scaled = lambda_xy * alpha * B_y
        B_z_scaled = lambda_z * alpha * B_z

        # Stack scaled B matrices
        B_matrix_scaled = vstack([B_x_scaled, B_y_scaled, B_z_scaled], format='csr')
        
        # Construct A_hat
        A_hat = vstack([A_matrix, B_matrix_scaled, mu * identity_matrix], format='csr')
        a_hat_sparsity = compute_sparsity(A_hat)
        # print(f'\t\tA_hat === Shape: {A_hat.shape}, Non-zero elements: {A_hat.nnz}, Sparsity: {a_hat_sparsity:.7f}')
        # print(f"\t\tA_hat sparsity: {a_hat_sparsity:.7f}")

        for direction in ['phs_x', 'phs_y', 'phs_z']:
            direction_index = ['phs_x', 'phs_y', 'phs_z'].index(direction)
            
            # Flatten E
            E_flattened_list = [all_data_unwrapped_voxel[subject_id][slice_num][direction][:, :, frame_number].flatten(order='F')
                                for slice_num in slices] # List of flattened E for each slice
            E_flattened = np.concatenate([E[~np.isnan(E)] for E in E_flattened_list]) # Concatenate non-NaN values only
            
            # Constructing zero matrix and L_prev
            zero_matrix_flattened = np.zeros((A_matrix.shape[1] * 3,)) # 3ngrid (if 3d) by 1
            
            if frame_number == 0:
                L_prev_flattened_nan = lagrangian_displacements_voxel[:, :, :, frame_number, direction_index].flatten(order='F')
            else:
                L_prev_flattened_nan = lagrangian_displacements_voxel[:, :, :, frame_number - 1, direction_index].flatten(order='F')
            
            # Replace NaNs with zeros
            L_prev_flattened = np.nan_to_num(L_prev_flattened_nan)  

            # Stack E_hat components
            E_hat = np.hstack([E_flattened, zero_matrix_flattened, mu * L_prev_flattened]).reshape(-1, 1)

                        
            # Solve the least squares problem
            result = spla.lsmr(A_hat, E_hat)
            # L_f_flattened = result[0]

            L_f_flattened, istop, itn, normr, normar, norma, conda, normx = result

            # print(f"\tDirection {direction}, Status: {istop}, Iterations: {itn},, conda: {conda:.2e} Residual: {normr:.2e}, n(ar): {normar:.2e}, n(a): {norma:.2e}, n(x): {normx:.2e}")
            # print(f"\t\t\tNorm of E_hat: {np.linalg.norm(E_hat)}")


            # Reshape and store Lagrangian displacement
            L_f = L_f_flattened.reshape((height, width, full_depth), order='F')
            lagrangian_displacements_voxel[:, :, :, frame_number, direction_index] = L_f

        print(f">> Frame {frame_number} processed in {time.time() - start_time:.2f} sec")
    
    # save the lagrangian displacements in mm as well
    lagrangian_displacements_mm = copy.deepcopy(lagrangian_displacements_voxel) # make a copy as pyhton numpy is MUTABLE
    lagrangian_displacements_mm[:,:,:, :, 0] = lagrangian_displacements_mm[:,:,:, :, 0] * voxel_size_x  # convert to mm for x direction
    lagrangian_displacements_mm[:,:,:, :, 1] = lagrangian_displacements_mm[:,:,:, :, 1] * voxel_size_y  # convert to mm for y direction
    lagrangian_displacements_mm[:,:,:, :, 2] = lagrangian_displacements_mm[:,:,:, :, 2] * voxel_size_z  # convert to mm for z direction

    print(f"\nUsing LSMR method, lambda={lambda_}, mu={mu}")
    print(f"Lagrangian displacement computed in {time.time() - start_time_all:.2f} sec")
    return lagrangian_displacements_voxel, all_data_unwrapped_voxel, lagrangian_displacements_mm, all_data_unwrapped_mm, num_bottom_slice, num_above_slice


