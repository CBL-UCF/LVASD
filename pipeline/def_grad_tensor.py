import numpy as np


def d_thin_plate_spline(r):
        """
        Derivative of thin plate spline radial basis function
        """
        # Avoid log of zero
        #r = np.where(r == 0, np.finfo(float).eps, r)
        return r * (1 + 2 * np.log(r))

def d_gaussian(r):
    """
    Derivative of gaussian radial basis function
    """
    return -2 * r * np.exp(-r**2)

def d_cubic(r):
    """
    Derivative of cubic radial basis function
    """
    return 3 * r**2

def d_linear(r):
    """
    Derivative of linear radial basis function
    """
    return -1

def d_quintic(r):
    """
    Derivative of quintic radial basis function
    """
    return -5 * r**4

def d_multiquadric(r):
    """
    Derivative of multiquadric radial basis function
    """
    return -r / np.sqrt(1 + r**2)

def d_inverse_multiquadric(r):
    """
    Derivative of inverse multiquadric radial basis function
    """
    return -2*r / (1 + r**2)**(3/2)

def d_inverse_quadratic(r):
    """
    Derivative of inverse quadratic radial basis function
    """
    return -2 / (1 + r**2)


d_kernel = {'thin_plate_spline':d_thin_plate_spline, 'gaussian':d_gaussian, 'cubic':d_cubic, 'linear':d_linear, 
            'quintic':d_quintic, 'multiquadric':d_multiquadric, 'inverse_multiquadric':d_inverse_multiquadric, 'inverse_quadratic':d_inverse_quadratic}


def compute_gradient(points, rbf_interpolators, kernel=None):
    """
    Compute the gradient of the RBF interpolators for each dimension at the given points.

    Parameters:
    - points: Array of shape (N, 2) or (N, 3) representing the points at which the gradient should be computed.
    - rbf_interpolators: List of RBFInterpolator objects, one for each output dimension.
    - kernel: The kernel to use for the RBF interpolation. Default is None.
    - disp: Boolean indicating if the gradient is the displacement gradient. Default is True.
    
    Returns:
    - The gradient matrix of the RBF interpolator at the given points.
    """
    if kernel is None:
        raise ValueError("Kernel not specified")
    
    kernel = kernel.lower()

    if kernel not in d_kernel.keys():
        raise ValueError("Kernel not supported")
    else:
        d_rbf = d_kernel[kernel]
    
    gradient_matrix = []
    num_dimensions = len(rbf_interpolators)
    point_dimensionality = points.shape[1]  # Determine if we're in 2D or 3D

    for dim in range(num_dimensions):
        rbf_interpolator = rbf_interpolators[dim]
        epsilon = rbf_interpolator.epsilon  # Dimension-specific epsilon
        scale = rbf_interpolator._scale     # Dimension-specific scale

        rbf_x = rbf_interpolator.y[:, 0] * epsilon
        rbf_y = rbf_interpolator.y[:, 1] * epsilon
        rbf_z = rbf_interpolator.y[:, 2] * epsilon if point_dimensionality == 3 else np.zeros_like(rbf_x)

        x = points[:, 0] * epsilon
        y = points[:, 1] * epsilon
        z = points[:, 2] * epsilon if point_dimensionality == 3 else np.zeros_like(x)

        distances = np.sqrt((x - rbf_x)**2 + (y - rbf_y)**2 + (z - rbf_z)**2)
        distances = np.where(distances == 0, np.finfo(float).eps, distances)

        rbf_coeffs = rbf_interpolator._coeffs[:len(rbf_interpolator.y)].squeeze()
        pol_coeffs = rbf_interpolator._coeffs[len(rbf_interpolator.y):].squeeze() if rbf_interpolator.powers.any() else None

        # Compute derivatives with respect to each coordinate for the current dimension
        d_dim_dx = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (x - rbf_x) / distances)
        d_dim_dy = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (y - rbf_y) / distances)
        d_dim_dz = np.sum(epsilon * rbf_coeffs * d_rbf(distances) * (z - rbf_z) / distances) if point_dimensionality == 3 else 0.0

        if pol_coeffs is not None:
            d_dim_dx += pol_coeffs[1] / scale[0]
            d_dim_dy += pol_coeffs[2] / scale[1]
            if point_dimensionality == 3:
                d_dim_dz += pol_coeffs[3] / scale[2]

        gradient_matrix.append([d_dim_dx, d_dim_dy, d_dim_dz])

    return np.array(gradient_matrix)

def inverse_gradient(grad):

    inv_grad = np.linalg.inv(grad)

    return inv_grad

def central_difference(f, x, h=1e-6):
    fx_plus_h = f(x + h)
    fx_minus_h = f(x - h)
    return (fx_plus_h - fx_minus_h) / (2 * h)

def numerical_gradient(points, rbf_interpolators):
    """
    Compute the numerical gradient using central difference.

    Parameters:
    - points: Array of shape (N, 2) or (N, 3) representing the points at which the gradient should be computed.
    - rbf_interpolators: List of RBFInterpolator objects, one for each output dimension.
    
    Returns:
    - The numerical gradient matrix at the given points.
    """
    num_dimensions = len(rbf_interpolators)
    point_dimensionality = points.shape[1]  # Determine if we're in 2D or 3D
    gradients = []

    for dim in range(num_dimensions):
        rbf_interpolator = rbf_interpolators[dim]

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2] if point_dimensionality == 3 else None

        def f_x(x_val):
            input_points = np.column_stack((x_val, y, z)) if z is not None else np.column_stack((x_val, y))
            return rbf_interpolator(input_points)

        def f_y(y_val):
            input_points = np.column_stack((x, y_val, z)) if z is not None else np.column_stack((x, y_val))
            return rbf_interpolator(input_points)

        df_dx = central_difference(f_x, x)
        df_dy = central_difference(f_y, y)

        if point_dimensionality == 3:
            def f_z(z_val):
                input_points = np.column_stack((x, y, z_val))
                return rbf_interpolator(input_points)

            df_dz = central_difference(f_z, z)
            gradients.append([df_dx, df_dy, df_dz])
        else:
            gradients.append([df_dx, df_dy])

    return np.array(gradients)
    
