import numpy as np
import random
import matplotlib.pyplot as plt
from pipeline.compute_resting_frame import compute_resting_frame_using_splines


def plot_comparison_eulerian_lagrangian(all_data_unwrapped, lagrangian_displacements, subject_id, num_bottom_slice, slice_idx=0):
    directions = ['phs_x', 'phs_y', 'phs_z']
    direction_labels = ['X-direction', 'Y-direction', 'Z-direction']
    column_labels = ['Eulerian', 'Lagrangian (Masked)', 'Lagrangian (Unmasked)']
    
    rng_h = [20, 60]
    rng_v = [20, 60]
    
    slice_list = list(sorted(all_data_unwrapped[subject_id].keys()))
    slice_num = slice_list[slice_idx]

    # Resting Frame as Mask
    resting_mask, _, _, _, _ = compute_resting_frame_using_splines(all_data_unwrapped, subject_id, slice_num)
    resting_mask = resting_mask.astype(int)
    eulerian_mask = np.where(resting_mask == 1, False, True)  # Mask for Eulerian data
    eulerian_mask = eulerian_mask[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1]]
 
    plane_index = slice_idx + num_bottom_slice

    color_limits = {
        'phs_x': [-3, 3],
        'phs_y': [-3, 3],
        'phs_z': [-1, 1]
    }

    num_frames = lagrangian_displacements.shape[3]
    for t in range(num_frames):
        fig, axs = plt.subplots(3, 3, figsize=(14, 12))

        for idx, direction in enumerate(directions):
            color_lim = color_limits[direction]
            eulerian_data = np.ma.masked_invalid(all_data_unwrapped[subject_id][slice_num][direction][rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], t])
            lagrangian_data = lagrangian_displacements[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], plane_index, t, idx]
            lagrangian_masked = np.ma.masked_array(lagrangian_data, mask=eulerian_mask)
            
            x_ticks = np.arange(0, eulerian_data.shape[1], 5)
            y_ticks = np.arange(0, eulerian_data.shape[0], 5)
            
            for j, (data, label) in enumerate(zip([eulerian_data, lagrangian_masked, lagrangian_data], column_labels)):
                im = axs[idx, j].imshow(data, cmap='seismic', vmin=color_lim[1], vmax=color_lim[0])
                axs[idx, j].set_title(f'{label}' if direction == 'phs_x' else '', fontsize=18)
                axs[idx, j].set_xticks(x_ticks)
                axs[idx, j].set_yticks(y_ticks)
                axs[idx, j].grid(True)
                plt.colorbar(im, ax=axs[idx, j], ticks=np.linspace(color_lim[0], color_lim[1], 5))
                im.set_clim(color_lim[0], color_lim[1])
            
            axs[idx, 0].set_ylabel(direction_labels[idx], fontsize=14)
            
            print(f"{direction}\n      max: Eulerian: {np.max(eulerian_data):.3f}, Lagrangian Masked: {np.max(lagrangian_masked):.3f}")
            print(f"      min: Eulerian: {np.min(eulerian_data):.3f}, Lagrangian Masked: {np.min(lagrangian_masked):.3f}")

        plt.suptitle(f"Pipeline | Subject: {subject_id[-2:]} | Slice: {slice_num:02d} | Time-Step: ({t+1}/{num_frames})", fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_lagrangian_myocardium_tracking(all_data_unwrapped, lagrangian_displacements, subject_id, num_bottom_slice, slice_idx=0):
    """
    Visualizes the tracked myocardium points using Lagrangian displacement.
    Arrows indicate the movement of each myocardium point.

    Parameters:
    - all_data_unwrapped: Dictionary with unwrapped Eulerian data.
    - lagrangian_displacements: 5D numpy array containing computed Lagrangian displacements.
    - subject_id: The subject ID.
    - num_bottom_slice: Number of slices added below for padding.
    - slice_idx: Index of the slice to visualize (default=0).
    """
    slice_list = list(sorted(all_data_unwrapped[subject_id].keys()))
    slice_num = slice_list[slice_idx]

    # Resting Frame as Mask
    resting_mask, _, _, _, _ = compute_resting_frame_using_splines(all_data_unwrapped, subject_id, slice_num)
    resting_mask = resting_mask.astype(int)
    myocardium_mask = (resting_mask == 1)  # Extract myocardium region

    plane_index = slice_idx + num_bottom_slice
    num_frames = lagrangian_displacements.shape[3]

    for t in range(num_frames):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract Lagrangian displacement (masked)
        lagrangian_x = np.ma.masked_array(lagrangian_displacements[:, :, plane_index, t, 0], mask=~myocardium_mask)
        lagrangian_y = np.ma.masked_array(lagrangian_displacements[:, :, plane_index, t, 1], mask=~myocardium_mask)

        # Plot the masked Lagrangian image
        ax.imshow(lagrangian_x, cmap='seismic', origin='upper', alpha=0.8)

        # Overlay displacement arrows
        rows, cols = np.where(myocardium_mask)  # Get myocardium voxel indices
        for row, col in zip(rows, cols):
            end_x = col + lagrangian_x[row, col]
            end_y = row - lagrangian_y[row, col]
            ax.arrow(col, row, end_x - col, end_y - row, color='black', head_width=0.5, head_length=0.5)

        ax.set_title(f"Lagrangian Myocardium Tracking | Subject {subject_id[-2:]} | Slice {slice_num:02d} | Time-Step {t+1}/{num_frames}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def plot_comparison_eulerian_lagrangian_myocardium_arrow(all_data_unwrapped, lagrangian_displacements, subject_id, num_bottom_slice, slice_idx=0):
    directions = ['phs_x', 'phs_y', 'phs_z']
    direction_labels = ['X-direction', 'Y-direction', 'Z-direction']
    column_labels = ['Eulerian', 'Lagrangian (Masked)', 'Lagrangian (Unmasked)']
    
    rng_h = [25, 50]
    rng_v = [25, 50]
    
    slice_list = list(sorted(all_data_unwrapped[subject_id].keys()))
    slice_num = slice_list[slice_idx]

    # Resting Frame as Mask
    resting_mask, _, _, _, _ = compute_resting_frame_using_splines(all_data_unwrapped, subject_id, slice_num)
    resting_mask = resting_mask.astype(int)

    # Ensure mask matches cropped data size
    myocardium_mask = (resting_mask == 1)
    myocardium_mask = myocardium_mask[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1]]  # Crop it properly

    eulerian_mask = np.where(myocardium_mask, False, True)  # Mask for Eulerian data

    plane_index = slice_idx + num_bottom_slice

    color_limits = {
        'phs_x': [-3, 3],
        'phs_y': [-3, 3],
        'phs_z': [-1, 1]
    }

    num_frames = lagrangian_displacements.shape[3]
    for t in range(num_frames):
        fig, axs = plt.subplots(3, 3, figsize=(14, 12))

        for idx, direction in enumerate(directions):
            color_lim = color_limits[direction]
            eulerian_data = np.ma.masked_invalid(all_data_unwrapped[subject_id][slice_num][direction][rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], t])
            lagrangian_data = lagrangian_displacements[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], plane_index, t, idx]
            lagrangian_masked = np.ma.masked_array(lagrangian_data, mask=eulerian_mask)
            
            x_ticks = np.arange(0, eulerian_data.shape[1], 5)
            y_ticks = np.arange(0, eulerian_data.shape[0], 5)
            
            for j, (data, label) in enumerate(zip([eulerian_data, lagrangian_masked, lagrangian_data], column_labels)):
                im = axs[idx, j].imshow(data, cmap='seismic', vmin=color_lim[1], vmax=color_lim[0])
                axs[idx, j].set_title(f'{label}' if direction == 'phs_x' else '', fontsize=18)
                axs[idx, j].set_xticks(x_ticks)
                axs[idx, j].set_yticks(y_ticks)
                axs[idx, j].grid(True)
                plt.colorbar(im, ax=axs[idx, j], ticks=np.linspace(color_lim[0], color_lim[1], 5))
                im.set_clim(color_lim[0], color_lim[1])

                # Add arrows only in Lagrangian Masked subplot 
                if j == 1 and direction in ['phs_x', 'phs_y']:  # Only for X and Y directions
                    lagrangian_x = np.ma.masked_array(
                        lagrangian_displacements[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], plane_index, t, 0],
                        mask=~myocardium_mask
                    )
                    lagrangian_y = np.ma.masked_array(
                        lagrangian_displacements[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1], plane_index, t, 1],
                        mask=~myocardium_mask
                    )

                    rows, cols = np.where(myocardium_mask)  # Get myocardium voxel indices
                    for row, col in zip(rows, cols):
                        end_x = col + lagrangian_x[row, col]
                        end_y = row - lagrangian_y[row, col]
                        axs[idx, j].arrow(col, row, end_x - col, end_y - row, color='black', head_width=0.4, head_length=0.4)

            axs[idx, 0].set_ylabel(direction_labels[idx], fontsize=14)

        plt.suptitle(f"Pipeline | Subject: {subject_id[-2:]} | Slice: {slice_num:02d} | Time-Step: ({t+1}/{num_frames})", fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_comparison_eulerian_lagrangian_myocardium_path_tracker(
    all_data_unwrapped, lagrangian_displacements, subject_id, num_bottom_slice, slice_idx=0
):
    """
    Tracks and plots the cumulative path of arrowheads from Lagrangian displacement across all frames.
    The myocardium mask is used as the background.

    Parameters:
    - all_data_unwrapped: Dictionary with unwrapped Eulerian data.
    - lagrangian_displacements: 5D numpy array containing computed Lagrangian displacements.
    - subject_id: The subject ID.
    - num_bottom_slice: Number of slices added below for padding.
    - slice_idx: Index of the slice to visualize (default=0).
    """
    slice_list = list(sorted(all_data_unwrapped[subject_id].keys()))
    slice_num = slice_list[slice_idx]

    # Resting Frame as Mask
    resting_mask, _, _, _, _ = compute_resting_frame_using_splines(all_data_unwrapped, subject_id, slice_num)
    resting_mask = resting_mask.astype(int)

    # Ensure mask matches cropped data size
    myocardium_mask = (resting_mask == 1)
    myocardium_mask = myocardium_mask[25:50, 25:50]  # Crop it properly

    plane_index = slice_idx + num_bottom_slice
    num_frames = lagrangian_displacements.shape[3]

    # Dictionary to store trajectories
    trajectory_dict = {}

    # Get myocardium voxel indices
    rows, cols = np.where(myocardium_mask)
    voxel_indices = list(zip(rows, cols))  # Convert to list for selection

    # Initialize paths for each myocardium voxel
    for row, col in voxel_indices:
        trajectory_dict[(row, col)] = [(col, row)]  # Start with initial position

    # Accumulate displacements over frames
    for t in range(num_frames):
        lagrangian_x = np.ma.masked_array(
            lagrangian_displacements[25:50, 25:50, plane_index, t, 0], mask=~myocardium_mask
        )
        lagrangian_y = np.ma.masked_array(
            lagrangian_displacements[25:50, 25:50, plane_index, t, 1], mask=~myocardium_mask
        )

        for row, col in zip(rows, cols):
            new_x = col + lagrangian_x[row, col]
            new_y = row - lagrangian_y[row, col]  # Negative for flipped coordinates
            trajectory_dict[(row, col)].append((new_x, new_y))

    # Randomly select 
    selected_paths = random.sample(voxel_indices, min(5, len(voxel_indices))) # Select 5 paths randomly

    # Create final plot
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Plot myocardium mask
    ax.imshow(myocardium_mask, cmap='gray_r')  # Background mask
    
    # Plot each trajectory
    for (row, col), path in trajectory_dict.items():
        path_x, path_y = zip(*path)  # Unpack path points
        color = 'red' if (row, col) in selected_paths else 'orange'  # Select color
        ax.plot(path_x, path_y, color=color, alpha=0.7, lw=2 if color == 'red' else 0.5)
    
    # Add tailless arrows every 10 frames
    for (row, col), path in trajectory_dict.items():
        color = 'red' if (row, col) in selected_paths else 'orange'
        for i in range(2, len(path) - 1, 10):  # Every 10th frame
            start_x, start_y = path[i]  
            end_x, end_y = path[i + 1]  
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                     color=color, head_width=0.3 if color == 'red' else 0.1, head_length=0.3 if color == 'red' else 0.1, 
                     alpha=0.8, length_includes_head=True)

    ax.set_title(f"Myocardium Path Tracker | Subject {subject_id[-2:]} | Slice {slice_num:02d}", fontsize=14)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect('equal')
    plt.show()


def plot_radial_circumferential_displacement(all_data_unwrapped_voxel, lagrangian_displacements_mm, subject_id, num_bottom_slice, slice_idx=0):
    """
    Computes and visualizes radial and circumferential displacement for each point.
    Uses a 2x2 subplot layout: 
        - First row: Radial displacement
        - Second row: Circumferential displacement
        - First column: Masked
        - Second column: Unmasked

    Parameters:
    - all_data_unwrapped_voxel: Dictionary with unwrapped Eulerian data as VOXEL should be used for the resting frame.
    - lagrangian_displacements: 5D numpy array containing computed Lagrangian displacements.
    - subject_id: The subject ID.
    - num_bottom_slice: Number of slices added below for padding.
    - slice_idx: Index of the slice to visualize (default=0).
    """
    rng_h = [25, 50]  # Define cropping range
    rng_v = [25, 50]

    slice_list = list(sorted(all_data_unwrapped_voxel[subject_id].keys()))
    slice_num = slice_list[slice_idx]

    # Resting Frame as Mask
    resting_mask, _, _, _, _ = compute_resting_frame_using_splines(all_data_unwrapped_voxel, subject_id, slice_num)
    resting_mask = resting_mask.astype(int)

    # Crop the myocardium mask first
    myocardium_mask = (resting_mask == 1)
    myocardium_mask = myocardium_mask[rng_h[0]:rng_h[1], rng_v[0]:rng_v[1]]

    # Compute centroid only from cropped region
    cropped_coords = np.column_stack(np.where(myocardium_mask == 1))
    centroid_row, centroid_col = np.mean(cropped_coords, axis=0)  # Compute centroid (y, x)

    plane_index = slice_idx + num_bottom_slice
    num_frames = lagrangian_displacements_mm.shape[3]

    # Loop through frames
    for t in range(num_frames):
        radial_displacement_matrix = np.zeros(myocardium_mask.shape)
        circumferential_displacement_matrix = np.zeros(myocardium_mask.shape)

        # Dictionary to store radial and circumferential vectors
        displacement_dict = {}

        for row in range(myocardium_mask.shape[0]):
            for col in range(myocardium_mask.shape[1]):
                # Compute radial unit vector (adjusted for cropped region)
                radial_vector = np.array([col - centroid_col, -(row - centroid_row)]) # The radial vector is from the centroid to the point
                radial_norm = np.linalg.norm(radial_vector)
                radial_unit_vector = radial_vector / radial_norm if radial_norm != 0 else np.array([0, 0])

                # Compute circumferential unit vector (perpendicular to radial vector)
                circumferential_unit_vector = np.array([-radial_unit_vector[1], radial_unit_vector[0]])  # Rotate 90 degrees

                # Get displacement vector
                displacement_vector = np.array([
                    lagrangian_displacements_mm[rng_h[0] + row, rng_v[0] + col, plane_index, t, 0],
                    lagrangian_displacements_mm[rng_h[0] + row, rng_v[0] + col, plane_index, t, 1] # Negate Y values as we are computing in physical coordinates 
                ])

                # Compute radial displacement (dot product)
                radial_displacement = np.dot(radial_unit_vector, displacement_vector)
                circumferential_displacement = np.dot(circumferential_unit_vector, displacement_vector)

                # Store results
                displacement_dict[(row, col)] = {
                    "radial_unit_vector": radial_unit_vector,
                    "circumferential_unit_vector": circumferential_unit_vector,
                    "displacement_vector": displacement_vector,
                    "radial_displacement": radial_displacement,
                    "circumferential_displacement": circumferential_displacement
                }

                # Store in matrices for visualization
                radial_displacement_matrix[row, col] = radial_displacement
                circumferential_displacement_matrix[row, col] = circumferential_displacement

        # Masked versions
        radial_masked = np.ma.masked_array(radial_displacement_matrix, mask=~myocardium_mask)
        circumferential_masked = np.ma.masked_array(circumferential_displacement_matrix, mask=~myocardium_mask)

        # Create 2x2 plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # First row: Radial displacement
        im1 = axs[0, 0].imshow(radial_masked, cmap="Blues_r", vmin=-14, vmax=0)
        axs[0, 0].scatter(centroid_col, centroid_row, color='yellow', s=100, edgecolor='black')
        axs[0, 0].set_title("Radial Displacement (Masked)")
        plt.colorbar(im1, ax=axs[0, 0])

        im2 = axs[0, 1].imshow(radial_displacement_matrix, cmap="Blues_r", vmin=-14, vmax=0)
        axs[0, 1].scatter(centroid_col, centroid_row, color='yellow', s=100, edgecolor='black')
        axs[0, 1].set_title("Radial Displacement (Unmasked)")
        plt.colorbar(im2, ax=axs[0, 1])

        # Second row: Circumferential displacement
        im3 = axs[1, 0].imshow(circumferential_masked, cmap="Blues_r", vmin=-10, vmax=0)
        axs[1, 0].scatter(centroid_col, centroid_row, color='yellow', s=100, edgecolor='black')
        axs[1, 0].set_title("Circumferential Displacement (Masked)")
        plt.colorbar(im3, ax=axs[1, 0])

        im4 = axs[1, 1].imshow(circumferential_displacement_matrix, cmap="Blues_r", vmin=-10, vmax=0)
        axs[1, 1].scatter(centroid_col, centroid_row, color='yellow', s=100, edgecolor='black')
        axs[1, 1].set_title("Circumferential Displacement (Unmasked)")
        plt.colorbar(im4, ax=axs[1, 1])

        # Adjust layout
        plt.suptitle(f"Radial & Circumferential Displacement (mm) | Subject {subject_id[-2:]} | Slice {slice_num:02d} | Frame {t+1}/{num_frames}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


