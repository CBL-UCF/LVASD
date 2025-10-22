import numpy as np
import matplotlib.pyplot as plt
from pipeline.def_grad_tensor import compute_gradient
from pipeline.rbf_interpolator import myRBFInterpolator
from pipeline.compute_resting_frame import compute_resting_frame_using_splines
from scipy.ndimage import binary_dilation
from matplotlib.path import Path


def compute_deformation_gradient(xyz_selected, rbf_interp, time_step, kernel="cubic"):
    """
    Computes the deformation gradient tensor F for selected points using RBF interpolation.
    """
    F_list = []
    for i in range(len(xyz_selected)):
        point = xyz_selected[i].reshape(1, -1)
        grad_matrix = compute_gradient(point, rbf_interp.access_rbf(time_step), kernel=kernel)
        F = np.eye(3) + grad_matrix
        F_list.append(F)
    return np.array(F_list)


def compute_strains_and_jacobian(xyz_selected, rbf_interp, time_step, kernel="cubic"):
    """
    Computes strain components (Err, Ecc, Ell) and Jacobian determinant (J) for selected voxels.
    """
    E_rr_list, E_cc_list, E_ll_list, J_list = [], [], [], []
    centroid = np.mean(xyz_selected, axis=0).reshape(1, -1)
    
    for i in range(len(xyz_selected)):
        point = xyz_selected[i].reshape(1, -1)
        F = compute_deformation_gradient(point, rbf_interp, time_step, kernel)[0]
        E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        
        radial_vector = (point - centroid) / np.linalg.norm(point - centroid)
        radial_vector = radial_vector.flatten()
        circumferential_vector = np.array([-radial_vector[1], radial_vector[0], 0])
        longitudinal_vector = np.array([0, 0, 1])
        
        E_rr = radial_vector.T @ E @ radial_vector
        E_cc = circumferential_vector.T @ E @ circumferential_vector
        E_ll = longitudinal_vector.T @ E @ longitudinal_vector
        J = np.linalg.det(F)
        
        E_rr_list.append(E_rr)
        E_cc_list.append(E_cc)
        E_ll_list.append(E_ll)
        J_list.append(J)
    
    return np.array(E_rr_list), np.array(E_cc_list), np.array(E_ll_list), np.array(J_list)


def plot_strains(x_selected, y_selected, E_rr_values, E_cc_values, E_ll_values, J_values):
    """
    Plots strain maps in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    size_marker = 140  
    font_size = 25
    FONT_SIZE = 20
    
    sc1 = axes[0, 0].scatter(x_selected, y_selected, c=J_values, cmap="seismic", s=size_marker, marker='s')
    axes[0, 0].set_title("Jacobian Determinant (J)", fontsize=font_size)
    fig.colorbar(sc1, ax=axes[0, 0]).ax.tick_params(labelsize=FONT_SIZE)
    
    sc2 = axes[0, 1].scatter(x_selected, y_selected, c=E_ll_values, cmap="seismic", s=size_marker, marker='s')
    axes[0, 1].set_title("Longitudinal Strain (E_ll)", fontsize=font_size)
    fig.colorbar(sc2, ax=axes[0, 1]).ax.tick_params(labelsize=FONT_SIZE)
    
    sc3 = axes[1, 0].scatter(x_selected, y_selected, c=E_cc_values, cmap="seismic", s=size_marker, marker='s')
    axes[1, 0].set_title("Circumferential Strain (E_cc)", fontsize=font_size)
    fig.colorbar(sc3, ax=axes[1, 0]).ax.tick_params(labelsize=FONT_SIZE)
    
    sc4 = axes[1, 1].scatter(x_selected, y_selected, c=E_rr_values, cmap="seismic", s=size_marker, marker='s')
    axes[1, 1].set_title("Radial Strain (E_rr)", fontsize=font_size)
    fig.colorbar(sc4, ax=axes[1, 1]).ax.tick_params(labelsize=FONT_SIZE)
    
    plt.tight_layout()
    plt.show()


def extract_voxel_info(lagrangian_displacements_voxel, all_data_unwrapped_voxel, lagrangian_displacements_mm, all_data_unwrapped_mm, subject_id, num_bottom_slice, num_above_slice, voxel_sizes, slice_locations):
    """
    Extracts myocardium voxel coordinates and Lagrangian displacements for strain computation in mm.
    Still Voxel based unwrapped data is used for the resting frame computation.

    Non myo strip of non-zero displacements added.
    """

    slice_list = list(sorted(all_data_unwrapped_voxel[subject_id].keys()))
    height, width, _, time_steps, _ = lagrangian_displacements_mm.shape

    voxel_size_x, voxel_size_y, voxel_size_z = voxel_sizes
    
    coordinates_all = [] # in mm
    displacements_all = [] # in mm

    epi_splines_all = []  # Store centered epicardium splines
    endo_splines_all = []  # Store centered endocardium splines

    centroids_all = [] # Store centroids of the myocardium for each slice
    resting_masks_all = [] # Store resting masks for all slices

    # Create a Z location starting from voxel size Z
    z_locations = [voxel_size_z * i for i in range(1, len(slice_list) + 1)] # in mm
    
    for slice_number in slice_list:
        
        # Resting Frame as Mask
        resting_mask, avg_endo, avg_epi, _, _, PS_frame = compute_resting_frame_using_splines(all_data_unwrapped_voxel, subject_id, slice_number) # For the resting frame the Eulerian displacement of VOXEL should be used
        resting_mask = resting_mask.astype(int)

        if slice_number == slice_list[0]:  # Only for the first slice
            # add equal number of num_bottom_slice slices with all zero to resting_masks_all
            for _ in range(num_bottom_slice):
                resting_masks_all.append(np.zeros((height, width), dtype=int))
        # Add the current resting mask
        resting_masks_all.append(resting_mask) # Store the resting mask
        if slice_number == slice_list[-1]:  # Only for the last slice
            # add equal number of num_above_slice slices with all zero to resting_masks_all
            for _ in range(num_above_slice):
                resting_masks_all.append(np.zeros((height, width), dtype=int))


        eulerian_mask = np.where(resting_mask == 1, True, False) # as we wanna keep where the mask is 1
        eulerian_mask_all = np.ones_like(eulerian_mask, dtype=bool) # Keep all points
        
        # Transform epicardium & endocardium splines to centered coordinates
        avg_endo_centered = np.zeros_like(avg_endo)
        avg_epi_centered = np.zeros_like(avg_epi)

        avg_endo_centered[:, 1] = - (avg_endo[:, 0] - ((height - 1) / 2)) * voxel_size_y  # Y to mm (centered) - as in the compute_resting_frame.py the first column is Y
        avg_endo_centered[:, 0] = (avg_endo[:, 1] - ((width - 1) / 2)) * voxel_size_x  # X to mm (centered) - as in the compute_resting_frame.py the second column is X

        avg_epi_centered[:, 1] = - (avg_epi[:, 0] - ((height - 1) / 2)) * voxel_size_y  # Y to mm (centered) - as in the compute_resting_frame.py the first column is Y
        avg_epi_centered[:, 0] = (avg_epi[:, 1] - ((width - 1) / 2)) * voxel_size_x  # X to mm (centered) - as in the compute_resting_frame.py the second column is X

        # Store transformed splines
        endo_splines_all.append(avg_endo_centered)
        epi_splines_all.append(avg_epi_centered)


        # Get the slice index and find the corresponding plane index in lagrangian_displacements_mm
        slice_idx = slice_list.index(slice_number)
        pln_idx = num_bottom_slice + slice_idx
        
        # Get the coordinates of the voxels in the Eulerian mask in mm (and centered)
        y_location_myo, x_location_myo = np.where(eulerian_mask)
        y_location_myo_centered = - (y_location_myo - ((height - 1) / 2)) * voxel_size_y # Turned to the center of the image and in mm (Not in voxel)
        x_location_myo_centered = (x_location_myo - ((width - 1) / 2)) * voxel_size_x # Turned to the center of the image and in mm (Not in voxel)
        
        # compute the centroid of the myocardium for each slice
        centroid = [np.mean(x_location_myo_centered), np.mean(y_location_myo_centered)]
        centroids_all.append(centroid)

        # all points
        structure = np.ones((5, 5))
        eulerian_mask_expanded = binary_dilation(eulerian_mask, structure=structure)

        y_location, x_location = np.where(eulerian_mask_expanded)
        y_location_centered = - (y_location - ((height - 1) / 2)) * voxel_size_y # Turned to the center of the image and in mm (Not in voxel)
        x_location_centered = (x_location - ((width - 1) / 2)) * voxel_size_x # Turned to the center of the image and in mm (Not in voxel)

        for point in range(len(y_location)):
            coordinates_all.append([x_location_centered[point], y_location_centered[point], z_locations[slice_idx]])
            displacements_time = []
            for t in range(time_steps):
                displacements = lagrangian_displacements_mm[y_location[point], x_location[point], pln_idx, t, :]
                displacements_time.append(displacements)
            displacements_all.append(displacements_time)
    
    coordinates_all = np.array(coordinates_all)         # in mm and Shape (N, 3)
    displacements_all = np.array(displacements_all)     # in mm and Shape (N, Time_steps, 3)
    displacements_all = np.transpose(displacements_all, (0, 2, 1))      # Shape from (N, Time_steps, 3) to (N, 3, Time_steps) for RBF interpolation
    
    endo_splines_all = np.array(endo_splines_all)  # Shape (num_slices, num_points, 2)
    epi_splines_all = np.array(epi_splines_all)  # Shape (num_slices, num_points, 2)

    centroids_all = np.array(centroids_all)  # Shape (num_slices, 2)
    resting_masks_all = np.array(resting_masks_all)  # Shape (num_slices, height, width)
    resting_masks_all = np.transpose(resting_masks_all, (1, 2, 0))  # Shape (height, width, num_slices)

    return coordinates_all, displacements_all, endo_splines_all, epi_splines_all, centroids_all, PS_frame, resting_masks_all


