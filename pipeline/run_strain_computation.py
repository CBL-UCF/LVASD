from pipeline.compute_lagrangian_displacement import compute_lagrangian_displacement 
from pipeline.compute_strains import extract_voxel_info
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pipeline.rbf_interpolator import myRBFInterpolator
from pipeline.def_grad_tensor import compute_gradient
from scipy.spatial import cKDTree
from matplotlib.path import Path
import pandas as pd
from pipeline.json_reader import read_crop_coords, read_encoding_frequencies, read_voxel_sizes, check_consistency, read_slice_locations
import seaborn as sns
from pipeline.vtk_exporter import export_slice_motion_to_vtk
from openpyxl import load_workbook


def generate_query_points(endo_spline, epi_spline, centroid, num_segments=5, angle_step=10):
    """
    Generate query points along radial lines from the centroid of the myocardium.

    Parameters:
    - endo_spline (np.ndarray): Endocardial spline points (N x 2).
    - epi_spline (np.ndarray): Epicardial spline points (N x 2).
    - num_segments (int): Number of segments to divide each radial line.
    - angle_step (int): Angle step in degrees for generating radial lines.

    Returns:
    - query_points (np.ndarray): All query points (M x 2).
    - endo_points (np.ndarray): Endocardial query points (K x 2).
    - mid_points (np.ndarray): Mid-wall query points (K x 2).
    - epi_points (np.ndarray): Epicardial query points (K x 2).
    """

    query_points = []
    endo_points = []
    mid_points = []
    epi_points = []

    for angle in range(0, 360, angle_step):
        theta = np.radians(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        endo_intersection = find_intersection(centroid, direction, endo_spline)
        epi_intersection = find_intersection(centroid, direction, epi_spline)

        if endo_intersection is not None and epi_intersection is not None:
            segment_points = np.linspace(endo_intersection, epi_intersection, num_segments + 1)
            mid_idx = num_segments // 2  # Only valid for odd num_segments

            for i in range(num_segments):
                midpoint = (segment_points[i] + segment_points[i + 1]) / 2
                query_points.append(midpoint)
                if num_segments % 2 == 1:
                    if i == 0:
                        endo_points.append(midpoint)
                    elif i == mid_idx:
                        mid_points.append(midpoint)
                    elif i == num_segments - 1:
                        epi_points.append(midpoint)

    return np.array(query_points), np.array(endo_points), np.array(mid_points), np.array(epi_points)


def find_intersection(centroid, direction, spline):
    """
    Find the intersection of a radial line with a spline.

    Parameters:
    - centroid (np.ndarray): The starting point of the radial line (2,).
    - direction (np.ndarray): The direction vector of the radial line (2,).
    - spline (np.ndarray): The spline points (N x 2).

    Returns:
    - intersection (np.ndarray or None): Intersection point (2,) or None if no intersection.
    """
    for i in range(len(spline) - 1):
        # Define the segment of the spline
        p1, p2 = spline[i], spline[i + 1]

        # Solve for intersection between the radial line and the spline segment
        A = np.array([direction, p1 - p2]).T
        b = p1 - centroid
        try:
            t, u = np.linalg.solve(A, b)
            if 0 <= u <= 1 and t >= 0:
                # Intersection point
                return centroid + t * direction
        except np.linalg.LinAlgError:
            # Lines are parallel, no intersection
            continue
    return None


def run_strain_computation(subject_id, lambda_=4, mu=0.05, crop_buffer= 0.08, crop_condition=True, save_unwrap=True):
    """
    Run the strain computation pipeline for a given subject ID.
    
    Parameters:
    - subject_id (str): The subject ID for which to compute strains.
    
    Returns:
    - None
    """
    
    # Step 1: Define the path to the NIFTI files of the subject
    nifti_path = f"results/segmentation/nifti/{subject_id}"

    # Subject name is derived from the volume ID
    # e.g., Vol_015 -> subject015
    subject_name = subject_id.replace("Vol_", "subject")

    # Step 2: Ensure the JSON files are consistent across slices
    check_consistency(subject_id)

    # Step 3: Read crop coordinates, encoding frequencies, and voxel sizes
    # crop_coords = read_crop_coords(subject_id)
    encoding_frequencies = read_encoding_frequencies(subject_id)
    voxel_sizes = read_voxel_sizes(subject_id)
    slice_locations = read_slice_locations(subject_id)

    # Step 4: Compute Lagrangian Displacement
    lagrangian_displacements_voxel, all_data_unwrapped_voxel, lagrangian_displacements_mm, all_data_unwrapped_mm, num_bottom_slice, num_above_slice = compute_lagrangian_displacement(
        base_directory = nifti_path, 
        subject_id = subject_name, 
        encoding_frequencies= encoding_frequencies, 
        voxel_sizes = voxel_sizes, 
        lambda_=lambda_, 
        mu=mu,
        crop_buffer=crop_buffer,
        crop_condition=crop_condition,
        save_unwrap=save_unwrap
    )

    # Step 5: Find the Myocardium Voxels (This gives the coords and displacements of the myocardium in mm)
    coordinates_all, displacements_all, endo_splines_all, epi_splines_all, centroids_all, PS_frame, resting_masks_all = extract_voxel_info(
    lagrangian_displacements_voxel, 
    all_data_unwrapped_voxel, 
    lagrangian_displacements_mm, 
    all_data_unwrapped_mm, 
    subject_name, 
    num_bottom_slice, 
    num_above_slice, 
    voxel_sizes,
    slice_locations
    )

    # Step 5.1: Save the coordinates and displacements as vtk files
    # Define the directory to save the VTK files
    vtk_save_dir = f'results/displacement/{subject_id}/vtk'
    # Clear the directory if it exists
    if os.path.exists(vtk_save_dir):
        shutil.rmtree(vtk_save_dir)
    os.makedirs(vtk_save_dir, exist_ok=True)


    # Step 5.2: Save the resting masks as npy files (for visualization purposes)
    resting_masks_dir = f'results/resting_masks/{subject_id}'
    if os.path.exists(resting_masks_dir):
        shutil.rmtree(resting_masks_dir)
    os.makedirs(resting_masks_dir, exist_ok=True)
    # save all slices resting masks in one npy file
    np.save(os.path.join(resting_masks_dir, 'resting_masks.npy'), np.array(resting_masks_all))

    # Step 5.3: Save the lagrangian displacements mm as npy files (for visualization purposes)
    lagrangian_disp_dir = f'results/displacement/{subject_id}/npy'
    if os.path.exists(lagrangian_disp_dir):
        shutil.rmtree(lagrangian_disp_dir)
    os.makedirs(lagrangian_disp_dir, exist_ok=True)
    # save all slices lagrangian displacements mm in one npy file
    np.save(os.path.join(lagrangian_disp_dir, 'lagrangian_displacements_mm.npy'), lagrangian_displacements_mm)


    ## Create the directory for saving query points vtk displacement files
    displacement_query_vtk_dir = f"results/displacement_query/{subject_id}/vtk"
    displacement_query_excel_dir = f"results/displacement_query/{subject_id}/excel"
    if os.path.exists(displacement_query_vtk_dir):
        shutil.rmtree(displacement_query_vtk_dir)
    if os.path.exists(displacement_query_excel_dir):
        shutil.rmtree(displacement_query_excel_dir)
    os.makedirs(displacement_query_vtk_dir, exist_ok=True)
    os.makedirs(displacement_query_excel_dir, exist_ok=True)


    # Step 6: Compute the Strains (Global RBF)
    kernel='cubic'
    print("\nFitting Global RBF Interpolator...\n")
    rbf_global = myRBFInterpolator(coordinates_all, displacements_all, kernel=kernel, epsilon=1, 
                                    smoothing=0.01, degree=None)

    # Define the path for saving the excel file
    strain_path_subject = f'results/strain/{subject_id}'
    # Clear the directory if it exists
    if os.path.exists(strain_path_subject):
        shutil.rmtree(strain_path_subject)
    os.makedirs(strain_path_subject)

    # Create Excel writers for Peak Systole and Median strain values
    strain_writer = pd.ExcelWriter(os.path.join(strain_path_subject, 'peak_systole_strain.xlsx'), engine='xlsxwriter')
    # Create Excel writer for Median strain values
    median_writer = pd.ExcelWriter(os.path.join(strain_path_subject, 'median_strain.xlsx'), engine='xlsxwriter')
    # Create a subfolder for saving strain values for all frames
    all_frames_subdir = os.path.join(strain_path_subject, 'all_frames')
    os.makedirs(all_frames_subdir, exist_ok=True)

    print()

    slice_list = list(sorted(all_data_unwrapped_voxel[subject_name].keys()))
    for slice_index in range(len(slice_list)):
        
        # Extract the coordinates of the myocardium voxels for the specified slice
        z_target = voxel_sizes[2] * (slice_index + 1)  # Z slice location
        mask = np.isclose(coordinates_all[:, 2], z_target, atol=1e-3)  # Find matching Z values
        x_selected = coordinates_all[mask, 0]
        y_selected = coordinates_all[mask, 1]
        points_for_specific_slice = np.column_stack((x_selected, y_selected, np.full_like(x_selected, z_target)))

        # spline pont
        endo_spline_selected = endo_splines_all[slice_index]
        epi_spline_selected = epi_splines_all[slice_index]
        # Concatenate endo_spline_selected and a reversed epi_spline_selected to create a closed polygon.
        polygon_points = np.concatenate((endo_spline_selected, epi_spline_selected[::-1]), axis=0)
        poly_path = Path(polygon_points) 

        # to save the vtk files for the slice
        inside_mask_original_points = poly_path.contains_points(points_for_specific_slice[:, :2])
        coord_slice = points_for_specific_slice[inside_mask_original_points]
        disp_slice = displacements_all[mask][inside_mask_original_points]

        # Save the slice motion to VTK (Old method - using original voxel centers) # Without the cylindrical mesh
        export_slice_motion_to_vtk(coord_slice, disp_slice, slice_list[slice_index], vtk_save_dir)

        # New queri points (by generating radial lines from the centroid of the myocardium)
        centroid = centroids_all[slice_index]  # shape (2,)
        endo_spline = endo_splines_all[slice_index]
        epi_spline = epi_splines_all[slice_index]

        radial_query_points_all, radial_query_points_endo, radial_query_points_mid, radial_query_points_epi = generate_query_points(
            endo_spline=endo_spline,
            epi_spline=epi_spline,
            centroid=centroid,
            num_segments=5,     # To get the mid-wall points use num_segments=1
            angle_step=5
        )

        # Form full 3D query points by adding correct z-slice (use same z_target)
        queri_points = np.column_stack((radial_query_points_all, np.full((len(radial_query_points_all),), z_target)))

        # Initialize a dictionary to hold global strain results
        global_strain_results = {}


        # Create a mask for the query points that are inside the polygon
        inside_mask = poly_path.contains_points(queri_points[:, :2])

        # Get number of query points and frames
        num_query_pts = queri_points.shape[0]
        
        # Get number of frames from lagrangian_displacements_voxel shape
        num_frames = lagrangian_displacements_voxel.shape[3] # (height, width, depth, time_steps, 3)
        time_steps = [i for i in range(num_frames)]

        J_median_global_slice = []
        E_rr_median_global_slice = []
        E_cc_median_global_slice = []
        E_ll_median_global_slice = []

        # Initialize displacement array at query points
        disp_query = np.zeros((num_query_pts, 3, num_frames))

        for time_step in time_steps:

            rbf_t = rbf_global.access_rbf(time_step)
            # disp_query[:, :, time_step] = rbf_t(queri_points)
            disp_query[:, 0, time_step] = rbf_t[0](queri_points).flatten()
            disp_query[:, 1, time_step] = rbf_t[1](queri_points).flatten()
            disp_query[:, 2, time_step] = rbf_t[2](queri_points).flatten()

            # ---------------------------------------------------------
            # Save displacement values at query points to Excel (per frame)
            # ---------------------------------------------------------
            disp_query_flat = disp_query[:, :, time_step]
            disp_query_df = pd.DataFrame({
                "x": queri_points[:, 0],
                "y": queri_points[:, 1],
                "z": queri_points[:, 2],
                "u": disp_query_flat[:, 0],
                "v": disp_query_flat[:, 1],
                "w": disp_query_flat[:, 2],
            })

            frame_excel_path = os.path.join(displacement_query_excel_dir, f"frame_{time_step+1}.xlsx")

            if os.path.exists(frame_excel_path):
                with pd.ExcelWriter(frame_excel_path, engine="openpyxl", mode="a") as writer:
                    disp_query_df.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)
            else:
                with pd.ExcelWriter(frame_excel_path, engine="openpyxl", mode="w") as writer:
                    disp_query_df.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)

            # Compute global strains
            global_strain_results["x"] = queri_points[:, 0]
            global_strain_results["y"] = queri_points[:, 1]
            global_strain_results["J"] = []
            global_strain_results["E_rr"] = []
            global_strain_results["E_cc"] = []
            global_strain_results["E_ll"] = []

            centroid_x, centroid_y = centroids_all[slice_index]
            centroid = np.array([centroid_x, centroid_y, 0])

            for i in range(len(queri_points)):
                point = queri_points[i].flatten()  # Ensures it's a 1D array (shape: (3,))
                
                grad_matrix = compute_gradient(point.reshape(1, -1), rbf_global.access_rbf(time_step), kernel=kernel)
                F = np.eye(3) + grad_matrix
                E = 0.5 * (np.dot(F.T, F) - np.eye(3))
                J = np.linalg.det(F)

                # Ensure radial_vector is computed correctly
                radial_vector = centroid - np.array([point[0], point[1], 0])  # Extract x, y correctly
                radial_vector /= np.linalg.norm(radial_vector)  # Normalize

                circumferential_vector = np.array([-radial_vector[1], radial_vector[0], 0])  # Perpendicular vector
                longitudinal_vector = np.array([0, 0, 1])  # Fixed longitudinal direction

                E_rr = radial_vector.T @ E @ radial_vector
                E_cc = circumferential_vector.T @ E @ circumferential_vector
                E_ll = longitudinal_vector.T @ E @ longitudinal_vector

                global_strain_results["J"].append(J)
                global_strain_results["E_rr"].append(E_rr)
                global_strain_results["E_cc"].append(E_cc)
                global_strain_results["E_ll"].append(E_ll)
            
            # Convert lists to numpy arrays
            for key in ["J", "E_rr", "E_cc", "E_ll"]:
                global_strain_results[key] = np.array(global_strain_results[key])
            
            # Save strain values for current frame to a separate Excel file
            strain_data_current_frame = {
                "x": global_strain_results["x"][inside_mask],
                "y": global_strain_results["y"][inside_mask],
                "J": global_strain_results["J"][inside_mask],
                "E_rr": global_strain_results["E_rr"][inside_mask],
                "E_cc": global_strain_results["E_cc"][inside_mask],
                "E_ll": global_strain_results["E_ll"][inside_mask]
            }
            # Create a DataFrame and save to Excel
            strain_df_all_frames = pd.DataFrame(strain_data_current_frame)
            # Save to Excel (one file per frame)
            frame_filename = os.path.join(all_frames_subdir, f"frame_{time_step+1}.xlsx")
            # with pd.ExcelWriter(frame_filename, engine='xlsxwriter') as writer:
            #     strain_df_all_frames.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)

            # Check if the file already exists
            if os.path.exists(frame_filename):
                # Load the existing workbook
                with pd.ExcelWriter(frame_filename, engine='openpyxl', mode='a') as writer:
                    # Append the current slice's data to a new sheet
                    strain_df_all_frames.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)
            else:
                # Create a new workbook and write the first sheet
                with pd.ExcelWriter(frame_filename, engine='openpyxl', mode='w') as writer:
                    strain_df_all_frames.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)


            #################################################### Save the epi, endo, and mid-wall strains ###############

            # --- Create subfolders for segmental strain maps (do this once, before the time loop) ---
            endo_subdir = os.path.join(all_frames_subdir, "endo")
            mid_subdir = os.path.join(all_frames_subdir, "mid")
            epi_subdir = os.path.join(all_frames_subdir, "epi")
            os.makedirs(endo_subdir, exist_ok=True)
            os.makedirs(mid_subdir, exist_ok=True)
            os.makedirs(epi_subdir, exist_ok=True)

            # --- Inside the time loop, after saving the full set ---
            # Build masks for endo/mid/epi points
            endo_mask = np.isin(queri_points[:, :2], radial_query_points_endo).all(axis=1) & inside_mask
            mid_mask = np.isin(queri_points[:, :2], radial_query_points_mid).all(axis=1) & inside_mask
            epi_mask = np.isin(queri_points[:, :2], radial_query_points_epi).all(axis=1) & inside_mask

            for seg_name, seg_mask, seg_subdir in zip(
                ["endo", "mid", "epi"],
                [endo_mask, mid_mask, epi_mask],
                [endo_subdir, mid_subdir, epi_subdir]
            ):
                strain_data_segment = {
                    "x": global_strain_results["x"][seg_mask],
                    "y": global_strain_results["y"][seg_mask],
                    "J": global_strain_results["J"][seg_mask],
                    "E_rr": global_strain_results["E_rr"][seg_mask],
                    "E_cc": global_strain_results["E_cc"][seg_mask],
                    "E_ll": global_strain_results["E_ll"][seg_mask]
                }
                strain_df_segment = pd.DataFrame(strain_data_segment)
                frame_filename_segment = os.path.join(seg_subdir, f"frame_{time_step+1}.xlsx")
                if os.path.exists(frame_filename_segment):
                    with pd.ExcelWriter(frame_filename_segment, engine='openpyxl', mode='a') as writer:
                        strain_df_segment.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)
                else:
                    with pd.ExcelWriter(frame_filename_segment, engine='openpyxl', mode='w') as writer:
                        strain_df_segment.to_excel(writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)            


            ####################################################### End of Saving epi, endo, and mid-wall strains ###########

            
            # If current time step is Peak Systole, store the strain values
            if time_step == PS_frame:
                # Make a copy to be used after the time loop
                strain_results_PS = { key: global_strain_results[key].copy() for key in global_strain_results }

            # Store the frame median value (inside the myocardium)
            J_median_global_frame = np.median(global_strain_results["J"][inside_mask])
            E_rr_median_global_frame = np.median(global_strain_results["E_rr"][inside_mask])
            E_cc_median_global_frame = np.median(global_strain_results["E_cc"][inside_mask])
            E_ll_median_global_frame = np.median(global_strain_results["E_ll"][inside_mask])

            # Append the median values to the slice lists
            J_median_global_slice.append(J_median_global_frame)
            E_rr_median_global_slice.append(E_rr_median_global_frame)
            E_cc_median_global_slice.append(E_cc_median_global_frame)
            E_ll_median_global_slice.append(E_ll_median_global_frame)

            print(f"Slice {slice_list[slice_index]} - Time Step {time_step} Done!")

        
        ######################### Export at the query points #############################

        export_slice_motion_to_vtk(
            coords=queri_points,
            disp=disp_query,
            slice_id=slice_list[slice_index],
            save_dir=displacement_query_vtk_dir
        )

        # Save the strain values from Peak Systole (if found)
        # creat the median and peak_systole subdirectories
        median_subdir = os.path.join(strain_path_subject, 'median')
        peak_systole_subdir = os.path.join(strain_path_subject, 'peak_systole')
        os.makedirs(median_subdir, exist_ok=True)
        os.makedirs(peak_systole_subdir, exist_ok=True)

        if strain_results_PS is not None:
            
            # Save the strain values for the slice (inside the myocardium)
            strain_data = {
            "x": strain_results_PS["x"][inside_mask],
            "y": strain_results_PS["y"][inside_mask],
            "J": strain_results_PS["J"][inside_mask],
            "E_rr": strain_results_PS["E_rr"][inside_mask],
            "E_cc": strain_results_PS["E_cc"][inside_mask],
            "E_ll": strain_results_PS["E_ll"][inside_mask]
            }

            strain_df = pd.DataFrame(strain_data)
            strain_df.to_excel(strain_writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)

            # Plot the strain map for the slice at Peak Systole
            strain_types = ["J", "E_rr", "E_cc", "E_ll"]
            strain_labels = ["Jacobian (J)", "$E_{rr}$", "$E_{cc}$", "$E_{ll}$"]
            for strain_type, strain_label in zip(strain_types, strain_labels):
                plt.figure(figsize=(8, 6))
                plt.scatter(strain_data["x"], strain_data["y"], c=strain_data[strain_type], cmap="viridis", s=100, marker='o')
                plt.colorbar(label=strain_label)
                plt.xlabel("X Coordinate", fontsize=12)
                plt.ylabel("Y Coordinate", fontsize=12)
                plt.title(f"{strain_label} - Peak Systole - Slice {slice_list[slice_index]}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save the plot
                plot_filename = f"{peak_systole_subdir}/{strain_type.lower()}_PS_slice{slice_list[slice_index]}.png"
                plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Peak Systole strain map saved: {plot_filename}")


        # Save median values for the slice (for all time steps)
        median_data = {
            "Time Step": time_steps,
            "J Median": J_median_global_slice,
            "E_rr Median": E_rr_median_global_slice,
            "E_cc Median": E_cc_median_global_slice,
            "E_ll Median": E_ll_median_global_slice
        }
        median_df = pd.DataFrame(median_data)
        median_df.to_excel(median_writer, sheet_name=f"Slice_{slice_list[slice_index]}", index=False)


    
    # Save the Excel files
    strain_writer.close()
    median_writer.close()

    print("Strain Computation - Done!")
    print()
    print("Strain Plots - Saving...")
    
    # Use Seaborn's built-in colorblind-friendly palette
    cb_palette = sns.color_palette("colorblind", 10)

    #read the median strain values from the Excel file
    median_file_path = os.path.join(strain_path_subject, 'median_strain.xlsx')

    # Load the Excel file and get all sheet names (each corresponding to a slice)
    median_excel = pd.ExcelFile(median_file_path)
    sheet_names = median_excel.sheet_names

    # Define Excel column names and corresponding plot labels with subscripts (LaTeX)
    excel_columns = ["J Median", "E_rr Median", "E_cc Median", "E_ll Median"]
    plot_labels = ["J Median", "$E_{rr}$ Median", "$E_{cc}$ Median", "$E_{ll}$ Median"]


    # Loop over each strain for plotting using its index and saving the plots
    for col, label in zip(excel_columns, plot_labels):
        plt.figure(figsize=(12,6))
        ax = plt.gca()
        ax.set_prop_cycle(color=cb_palette)  # set color cycle to our Seaborn palette

        # Loop over each sheet (each slice)
        for sheet in sheet_names:
            # Read data from the current sheet
            df = pd.read_excel(median_file_path, sheet_name=sheet)
            plt.plot(df["Time Step"], df[col], label=sheet, linewidth=2)

        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel(label, fontsize=14)
        plt.title(f"{label} - {subject_id} - (Global RBF)", fontsize=16, fontweight='bold')
        plt.legend(title="Slice", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        
        plt.tight_layout()
        plt.grid()
        plt.show()

        # Save the plot
        plot_filename = f"{median_subdir}/{col.replace(' Median', '_median').lower()}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"Plot saved: {plot_filename}")
    print("Strain Plots - Done!")


