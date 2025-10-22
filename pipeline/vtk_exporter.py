from pyvista import PolyData
import os
import numpy as np
import pyvista as pv


def export_slice_motion_to_vtk(coords, disp, slice_id, save_dir):
    """Save per-slice motion to .vtp files."""
    
    os.makedirs(save_dir, exist_ok=True)
    n_points, _, n_frames = disp.shape

    for t in range(n_frames):
        displaced = coords + disp[:, :, t]
        pdata = pv.PolyData(displaced)
        pdata["Frame"] = np.full((n_points,), t)
        fname = os.path.join(save_dir, f"slice{slice_id:02d}_frame_{t+1:02d}.vtp")
        pdata.save(fname)

