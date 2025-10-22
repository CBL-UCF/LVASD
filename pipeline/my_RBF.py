
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
import copy

# Define analytical derivatives of RBF kernels
def d_thin_plate_spline(r): return r * (1 + 2 * np.log(r))
def d_gaussian(r): return -2 * r * np.exp(-r**2)
def d_cubic(r): return 3 * r**2
def d_linear(r): return -1
def d_quintic(r): return -5 * r**4
def d_multiquadric(r): return -r / np.sqrt(1 + r**2)
def d_inverse_multiquadric(r): return -2 * r / (1 + r**2)**(3/2)
def d_inverse_quadratic(r): return -2 / (1 + r**2)

# Kernel derivative lookup
d_kernel = {
    'thin_plate_spline': d_thin_plate_spline,
    'gaussian': d_gaussian,
    'cubic': d_cubic,
    'linear': d_linear,
    'quintic': d_quintic,
    'multiquadric': d_multiquadric,
    'inverse_multiquadric': d_inverse_multiquadric,
    'inverse_quadratic': d_inverse_quadratic
}

class MyRBFInterpolator:
    def __init__(self, points, values, kernel='cubic', epsilon=1.0, smoothing=0.1):
        """
        Custom RBF Interpolator using SciPy's RBFInterpolator with analytical gradient computation.
        :param points: (N, 3) array of known 3D points.
        :param values: (N, 3, Time_steps) array of displacement values (Dx, Dy, Dz) per time step.
        :param kernel: RBF kernel type ('cubic', 'thin_plate_spline', 'gaussian', etc.).
        :param epsilon: Scaling parameter for certain RBFs.
        :param smoothing: Regularization parameter to smooth noisy data.
        """
        self.points = np.asarray(points)
        self.values = np.asarray(values)
        self.kernel = kernel.lower()
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.time_steps = self.values.shape[2]
        self.rbf_models = self._generate_rbfs()

    def _generate_rbfs(self):
        """Fit RBFInterpolator separately for each time step."""
        rbf_models = []
        for t in range(self.time_steps):
            rbf_models.append([
                RBFInterpolator(
                    self.points, 
                    self.values[:, dim, t], 
                    kernel=self.kernel, 
                    epsilon=self.epsilon, 
                    smoothing=self.smoothing, 
                    degree=None  # SciPy optimizes linear term
                ) for dim in range(3)
            ])
        return rbf_models

    def interpolate(self, new_points, time_step):
        """
        Interpolate displacement at new points for a specific time step.
        :param new_points: (M, 3) array of query points.
        :param time_step: Time step index.
        :return: (M, 3) array of interpolated displacements.
        """
        if time_step >= self.time_steps:
            raise ValueError(f"Invalid time_step {time_step}: Only {self.time_steps} available.")
        
        new_points = np.asarray(new_points)
        return np.column_stack([rbf(new_points) for rbf in self.rbf_models[time_step]])

    def compute_gradient(self, new_points, time_step):
        """
        Compute the gradient of the RBF interpolators for each dimension at the given points.

        Parameters:
        - new_points: Array of shape (N, 3) representing the points at which the gradient should be computed.
        - time_step: The time step index for which the gradient is computed.
        
        Returns:
        - The gradient matrix of the RBF interpolator at the given points.
        """
        if self.kernel not in d_kernel:
            raise ValueError(f"Kernel {self.kernel} not supported for analytical gradient computation.")
        
        d_rbf = d_kernel[self.kernel]  # Get the correct derivative function
        
        num_dimensions = len(self.rbf_models[time_step])
        gradient_matrix = []
        point_dimensionality = new_points.shape[1]  # 3D points

        for dim in range(num_dimensions):
            rbf_interpolator = self.rbf_models[time_step][dim]
            epsilon = rbf_interpolator.epsilon  # Dimension-specific epsilon
            scale = getattr(rbf_interpolator, '_scale', np.ones(3))  # Handle scale if present

            rbf_x = self.points[:, 0] * epsilon
            rbf_y = self.points[:, 1] * epsilon
            rbf_z = self.points[:, 2] * epsilon if point_dimensionality == 3 else np.zeros_like(rbf_x)

            x = new_points[:, 0] * epsilon
            y = new_points[:, 1] * epsilon
            z = new_points[:, 2] * epsilon if point_dimensionality == 3 else np.zeros_like(x)

            distances = np.sqrt((x[:, None] - rbf_x[None, :])**2 + 
                                (y[:, None] - rbf_y[None, :])**2 + 
                                (z[:, None] - rbf_z[None, :])**2)
            distances = np.where(distances == 0, np.finfo(float).eps, distances)

            rbf_coeffs = rbf_interpolator(self.points)  # Extract RBF weights

            # Compute derivatives
            d_dim_dx = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (x[:, None] - rbf_x[None, :]) / distances, axis=1)
            d_dim_dy = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (y[:, None] - rbf_y[None, :]) / distances, axis=1)
            d_dim_dz = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (z[:, None] - rbf_z[None, :]) / distances, axis=1) if point_dimensionality == 3 else np.zeros_like(d_dim_dx)

            gradient_matrix.append(np.column_stack((d_dim_dx, d_dim_dy, d_dim_dz)))

        return np.stack(gradient_matrix, axis=1)  # Shape (M, 3, 3)


