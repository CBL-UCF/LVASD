from scipy.interpolate import RBFInterpolator
import numpy as np
import copy


class myRBFInterpolator:
    def __init__(self, xyz, d, kernel='thin_plate_spline', epsilon=1, smoothing=0.01, degree=None):
        """
        Initialize the RBFInterpolator with input data and generate RBF functions for each phase and dimension.
        
        Parameters:
        - xyz: Array of shape (N, 2) or (N, 3) representing the input positions.
        - d: Array of shape (N, 3, P) representing the values for P phases.
        - kernel: The kernel to use for the RBF interpolation. Default is 'thin_plate_spline'.
        - epsilon: The epsilon parameter for the RBF interpolation. Default is 1.
        - smoothing: The smoothing parameter for the RBF interpolation. Default is 0.
        - degree: Degree of the added polynomial. For some RBFs the interpolant may not be well-posed if the polynomial degree is too small. 
                  Those RBFs and their corresponding minimum degrees are:
            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2
            The default value is the minimum degree for kernel or 0 if there is no minimum degree. Set this to -1 for no added polynomial.
        The parameters xyz and d are deep copied to preserve class internal state.
        """
        self.xyz = copy.deepcopy(xyz)
        self.d = copy.deepcopy(d)
        self.input_dim = self.xyz.shape[1]  # Store the dimensionality (2 or 3)
        self.rbf_list = self._generate_rbfs(kernel, epsilon, smoothing, degree)

    def _generate_rbfs(self, kernel, epsilon, smoothing, degree):
        """Generate RBF functions for each phase and each dimension."""
        rbfs = []
        for i in range(self.d.shape[2]):  # Loop over each phase
            print(f"Generating RBFs for time-step {i} ...")
            phase_rbfs = []
            for dim in range(self.d.shape[1]):  # Loop over each dimension
                rbf = RBFInterpolator(
                    self.xyz,
                    self.d[:, dim:dim+1, i],  # Only one dimension at a time
                    kernel=kernel,
                    epsilon=epsilon,
                    smoothing=smoothing,
                    degree=degree
                )
                phase_rbfs.append(rbf)
            rbfs.append(phase_rbfs)  # List of RBFs for each dimension in this phase
        return rbfs

    def access_rbf(self, phase_index, dim_index=None):
        """
        Access the RBF function(s) for the given phase and optional dimension index.

        Parameters:
        - phase_index: Index of the phase (0 to P-1) for which the RBFs should be used.
        - dim_index: Optional index of the dimension (0 to 2) for which the RBF should be used.
                     If None, all three dimension RBFs for the specified phase are returned.

        Returns:
        - The RBF function for the specified dimension, or a list of all three RBF functions if dim_index is None.
        """
        if phase_index < 0 or phase_index >= self.d.shape[2]:
            raise ValueError(f"Invalid phase index. It should be between 0 and {self.d.shape[2]-1}.")

        if dim_index is None:
            return self.rbf_list[phase_index]  # Return all three dimension RBFs for this phase
        elif dim_index < 0 or dim_index >= self.d.shape[1]:
            raise ValueError(f"Invalid dimension index. It should be between 0 and {self.d.shape[1]-1}.")
        
        return self.rbf_list[phase_index][dim_index]  # Return RBF for the specified dimension

    def predict(self, xyz, phase_index):
        """
        Predict the interpolated values for the given coordinates and phase index.
        
        Parameters:
        - xyz: Coordinates of shape (N, 2) or (N, 3) for which the prediction is sought.
        - phase_index: Index of the phase (0 to P-1) for which the RBFs should be used.
        
        Returns:
        - A tuple of interpolated values for the three dimensions.
        """
        if phase_index < 0 or phase_index >= self.d.shape[2]:
            raise ValueError(f"Invalid phase index. It should be between 0 and {self.d.shape[2]-1}.")
        
        if xyz.shape[1] != self.input_dim:
            raise ValueError(f"Input dimensionality mismatch. Expected {self.input_dim} dimensions.")

        # Apply each dimension's RBF interpolator
        phase_rbfs = self.rbf_list[phase_index]
        results = np.column_stack([rbf(xyz) for rbf in phase_rbfs])  # Collect results for each dimension
        return results
