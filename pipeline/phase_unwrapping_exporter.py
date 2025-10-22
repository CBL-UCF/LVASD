import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


def save_phase_unwrap_intermediates(all_data_cropped, all_data_unwrapped, subject_id):
    """
    Save 2x3 plots for each frame of each slice for a subject, showing wrapped and unwrapped phase data (X, Y, Z).

    Args:
        all_data_cropped (dict): Cropped data with wrapped phase ('phs_x', 'phs_y', 'phs_z').
        all_data_unwrapped (dict): Unwrapped phase data with same keys.
        subject_id (str): Subject ID to process.
        output_dir (str): Directory to save the plots.
    """

    subject_folder = "Vol_" + subject_id[-3:]  # Assuming subject_id is like "Vol_001"
    output_dir = f"results/phase_unwrapping/{subject_folder}"
    # clear the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # Iterate through slices and frames
    # get the global min and max across all frames and slices for consistent color scaling
    for slice_num in all_data_cropped[subject_id]:
        wrapped = all_data_cropped[subject_id][slice_num]
        unwrapped = all_data_unwrapped[subject_id][slice_num]
        num_frames = wrapped['phs_x'].shape[-1]
        for frame in range(num_frames):
            # Masked arrays: mask where unwrapped is nan
            wx = np.ma.masked_where(np.isnan(unwrapped['phs_x'][:, :, frame]), wrapped['phs_x'][:, :, frame])
            wy = np.ma.masked_where(np.isnan(unwrapped['phs_y'][:, :, frame]), wrapped['phs_y'][:, :, frame])
            wz = np.ma.masked_where(np.isnan(unwrapped['phs_z'][:, :, frame]), wrapped['phs_z'][:, :, frame])
            ux = np.ma.masked_where(np.isnan(unwrapped['phs_x'][:, :, frame]), unwrapped['phs_x'][:, :, frame])
            uy = np.ma.masked_where(np.isnan(unwrapped['phs_y'][:, :, frame]), unwrapped['phs_y'][:, :, frame])
            uz = np.ma.masked_where(np.isnan(unwrapped['phs_z'][:, :, frame]), unwrapped['phs_z'][:, :, frame])
            if frame == 0 and slice_num == list(all_data_cropped[subject_id].keys())[0]:
                global_min = min(wx.min(), wy.min(), wz.min(), ux.min(), uy.min(), uz.min())
                global_max = max(wx.max(), wy.max(), wz.max(), ux.max(), uy.max(), uz.max())
            else:
                global_min = min(global_min, wx.min(), wy.min(), wz.min(), ux.min(), uy.min(), uz.min())
                global_max = max(global_max, wx.max(), wy.max(), wz.max(), ux.max(), uy.max(), uz.max())
    
    downrange = - np.max(np.abs([global_min, global_max]))
    uprange = np.max(np.abs([global_min, global_max]))

    slices = all_data_cropped[subject_id]
    for slice_num in slices:
        wrapped = all_data_cropped[subject_id][slice_num]
        unwrapped = all_data_unwrapped[subject_id][slice_num]
        num_frames = wrapped['phs_x'].shape[-1]
        for frame in range(num_frames):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            # Masked arrays: mask where unwrapped is nan
            wx = np.ma.masked_where(np.isnan(unwrapped['phs_x'][:, :, frame]), wrapped['phs_x'][:, :, frame])
            wy = np.ma.masked_where(np.isnan(unwrapped['phs_y'][:, :, frame]), wrapped['phs_y'][:, :, frame])
            wz = np.ma.masked_where(np.isnan(unwrapped['phs_z'][:, :, frame]), wrapped['phs_z'][:, :, frame])
            ux = np.ma.masked_where(np.isnan(unwrapped['phs_x'][:, :, frame]), unwrapped['phs_x'][:, :, frame])
            uy = np.ma.masked_where(np.isnan(unwrapped['phs_y'][:, :, frame]), unwrapped['phs_y'][:, :, frame])
            uz = np.ma.masked_where(np.isnan(unwrapped['phs_z'][:, :, frame]), unwrapped['phs_z'][:, :, frame])
            # Wrapped
            im0 = axes[0, 0].imshow(wx, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[0, 0].set_title('Wrapped X')
            fig.colorbar(im0, ax=axes[0, 0])
            im1 = axes[0, 1].imshow(wy, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[0, 1].set_title('Wrapped Y')
            fig.colorbar(im1, ax=axes[0, 1])
            im2 = axes[0, 2].imshow(wz, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[0, 2].set_title('Wrapped Z')
            fig.colorbar(im2, ax=axes[0, 2])
            # Unwrapped
            im3 = axes[1, 0].imshow(ux, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[1, 0].set_title('Unwrapped X')
            fig.colorbar(im3, ax=axes[1, 0])
            im4 = axes[1, 1].imshow(uy, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[1, 1].set_title('Unwrapped Y')
            fig.colorbar(im4, ax=axes[1, 1])
            im5 = axes[1, 2].imshow(uz, cmap='RdBu', vmin=downrange, vmax=uprange)
            axes[1, 2].set_title('Unwrapped Z')
            fig.colorbar(im5, ax=axes[1, 2])
            for ax in axes.flat:
                ax.axis('off')
            base_name = f"sub{int(subject_id[-3:]):03d}_slc{int(slice_num):02d}_frm{frame+1:02d}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, base_name), bbox_inches="tight")
            plt.close(fig)