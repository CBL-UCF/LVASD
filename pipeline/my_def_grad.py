import numpy as np

def compute_gradient(query_points, rbf_interpolator, time_step):
    """
    Compute deformation gradient tensor F at given query points.
    
    :param query_points: (N, 3) array of points where gradient is computed.
    :param rbf_interpolator: Instance of MyRBFInterpolator.
    :param time_step: Time step index to use for interpolation.
    :return: (N, 3, 3) array of deformation gradient tensors.
    """
    N = len(query_points)
    grad_tensors = np.zeros((N, 3, 3))

    # Compute gradients for each point
    grad_tensors = rbf_interpolator.compute_gradient(query_points, time_step)

    return grad_tensors
