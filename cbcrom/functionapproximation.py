# Copyright (C) 2015 Benjamin Lackey
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import numpy as np
import scipy.optimize as optimize
import scipy.interpolate
import scipy.ndimage # interpolation on a uniform grid


def fit_generator(param_array, data_array, func):
    """
        Generate a function(params) that fits the data.
        
        Parameters
        ----------
        param_array : 2d array
        Parameter values.
        data_array : 1d array
        Function values at the corresponding parameter values.
        func(params, *args) : function to use for the fitting
        Params is 1d array of parameters.
        *args are the coefficients.
        
        Returns
        -------
        fit : function(params)
        A function constructed from func using the best fit parameters.
        """
    
    #     # curve_fit assumes that func can take an array of parameter values and return an array of function values
    #     # If func has conditional expressions, then this assumption will fail.
    #     # Use np.vectorize to fix this.
    #     func_wrapper = np.vectorize(func)
    #     coeffs, covariances = optimize.curve_fit(func_wrapper, param_array.T, data_array, maxfev=50000)
    coeffs, covariances = optimize.curve_fit(func, param_array.T, data_array, maxfev=250000)
    #print coeffs
    # Return a function
    def fit(params):
        return func(params, *coeffs)
    
    return fit


def uniform_grid_interpolation_generator(data, xmin, xmax, order=2):
    """
        Generate a function of N parameters that interpolates a uniformly spaced Nd array.
        
        Parameters
        ----------
        data : Nd array
        Data to interpolate.
        xmin : 1d array of length N
        Minimum value of each parameter.
        xmax : 1d array of length N
        Maximum value of each parameter.
        
        Returns
        -------
        
        interp: function(x)
        Interpolating function that takes x
        (1d array of the parameter values at which you want to interpolate the data).
        """
    
    # get number of points in each dimension
    Nx = np.array(data.shape)
    
    def interp(x):
        # rescale to pixel coordinates (0 to N-1 in each dimension)
        x_rescale = (x-xmin) * (Nx-1.0) / (xmax-xmin)
        
        # list of [[x], [y], [z], ...] coordinates for one point
        x_list = np.array([x_rescale]).T.tolist()
        
        # interpolate using the function that assumes pixel coordinates
        #return scipy.ndimage.map_coordinates(data, x_list, order=order, mode=np.nan)[0]
        return scipy.ndimage.map_coordinates(data, x_list, order=order, mode='constant', cval=np.nan)[0]
    #return scipy.ndimage.map_coordinates(data, x_list, order=order)[0]
    
    return interp


def unstructured_linear_interpolation_generator(param_array, data_array, nearest_outside=True):
    """
        Generate a linear interpolating function(params) from an unstructured grid.
        This is a wrapper for scipy.interpolate.LinearNDInterpolator.
        
        Parameters
        ----------
        param_array : 2d array
        Parameter values.
        data_array : 1d array
        Function values at the corresponding parameter values.
        nearest_outside : {bool, True}
        Do nearest neighor interpolation outside the convex hull if True.
        Return nan if False.
        
        Returns
        -------
        interp : function(params)
        An inteprolating function.
        """
    
    # This will return linear interpolation, but only in the convex hull
    linearinterp = scipy.interpolate.LinearNDInterpolator(param_array, data_array)
    # This will return the nearest neighbor
    nearestinterp = scipy.interpolate.NearestNDInterpolator(param_array, data_array)
    
    # Return a function that chooses between linear and nearest neighbor interpolation
    def interp(params):
        linval = linearinterp(params)[0]
        if np.isnan(linval) and nearest_outside:
            return nearestinterp(params)[0]
        else:
            return linval

    return interp


def amp_function_generator_from_list(param_array, amp_array):
    """
        Generate a function to fit and/or interpolate the amplitude from a list of values.
        Parameters
        ----------
        param_array : 2d arraylike
        Array of the parameters at each point in the list.
        amp_array : array
        Amplitude at each point in the list.
        
        Returns
        -------
        amp_function(params) : function
        The amplitude function.
        """
    
    # Do the fit
    return fit_generator(param_array, amp_array, rational_amp_fit)

# Do linear interpolation on nonuniform grid
#return unstructured_linear_interpolation_generator(param_array, amp_array)

# do both fit and interpolation
#     def amp_function(params):
#         rough_amp_fit_func = fit_generator(param_array, amp_array, rational_amp_fit)
#         rough_fit_array = np.array([rough_amp_fit_func(p) for p in param_array])
#         interp_function = unstructured_linear_interpolation_generator(param_array, amp_array/rough_fit_array)
#         return rough_amp_fit_func(params)*interp_function(params)
#     return amp_function


def phase_function_generator_from_list(param_array, phase_array):
    """
        Generate a function to fit and/or interpolate the phase from a list of values.
        Parameters
        ----------
        param_array : 2d arraylike
        Array of the parameters at each point in the list.
        phase_array : array
        Phase at each point in the list.
        
        Returns
        -------
        phase_function(params) : function
        The phase function.
        """
    
    #Do the fit
    return fit_generator(param_array, phase_array, rational_phase_fit)

# Do linear interpolation on nonuniform grid
#return unstructured_linear_interpolation_generator(param_array, phase_array)

# do both fit and interpolation
#     def phase_function(params):
#         rough_phase_fit_func = fit_generator(param_array, phase_array, rational_phase_fit)
#         rough_fit_array = np.array([rough_phase_fit_func(p) for p in param_array])
#         interp_function = unstructured_linear_interpolation_generator(param_array, phase_array/rough_fit_array)
#         return rough_phase_fit_func(params)*interp_function(params)
#     return phase_function


def amp_phase_functions_from_list_at_empirical_nodes(param_array, amp_at_nodes, phase_at_nodes):
    """
        Generate interpolating functions for amp and phase at each of the empirical_nodes T_i.
        
        Parameters
        ----------
        param_array : 2d arraylike
        Array of the parameters at each point in the list.
        amp_at_nodes : 2d array
        Amplitude at the empirical_nodes for each set of parameters.
        phase_at_nodes : 2d array
        Phase at the empirical_nodes for each set of parameters.
        
        Returns
        -------
        amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
        phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.
        """
    
    # Number of empirical nodes T_i
    Nnodes = amp_at_nodes.shape[-1]
    
    # Generate amp(params) function at each empirical node
    amp_function_list = []
    for nodei in range(Nnodes):
        print nodei,
        amp_array = amp_at_nodes[:, nodei]
        amp_function = amp_function_generator_from_list(param_array, amp_array)
        amp_function_list.append(amp_function)
    
    # Generate phase(params) function at each empirical node
    phase_function_list = []
    for nodei in range(Nnodes):
        print nodei,
        phase_array = phase_at_nodes[:, nodei]
        phase_function = phase_function_generator_from_list(param_array, phase_array)
        phase_function_list.append(phase_function)
    
    return amp_function_list, phase_function_list


# def amp_function_generator(param_grid, amp_grid):
#     """
#     Parameters
#     ----------
#     param_grid : (N+1)d array
#         The N Parameter values at each point on the parameter grid.
#     amp_grid : Nd array
#         Amplitude at each point on the parameter grid.

#     Returns
#     -------
#     amp_function(params) : function
#         The amplitude function that passes through the amp_grid data at the points in param_grid.
#     """

#     # Flatten param_grid and amp_grid for the fitting function
#     Nparams = param_grid.shape[-1]
#     Npoints = product(list(param_grid.shape)[:-1])
#     param_array = param_grid.reshape((Npoints, Nparams))
#     amp_array = amp_grid.flatten()

#     # Do the fit
#     return fit_generator(param_array, amp_array, rational_amp_fit)


# def amp_function_generator(param_grid, amp_grid):
#     """
#     Parameters
#     ----------
#     param_grid : (N+1)d array
#         The N Parameter values at each point on the parameter grid.
#     amp_grid : Nd array
#         Amplitude at each point on the parameter grid.

#     Returns
#     -------
#     amp_function(params) : function
#         The amplitude function that passes through the amp_grid data at the points in param_grid.
#     """

#     # Flatten param_grid and amp_grid for the fitting function
#     Nparams = param_grid.shape[-1]
#     params_min = param_grid[tuple([0]*Nparams)]
#     params_max = param_grid[tuple([-1]*Nparams)]

#     Npoints = product(list(param_grid.shape)[:-1])
#     param_array = param_grid.reshape((Npoints, Nparams))
#     amp_array = amp_grid.flatten()

#     # Do the fit
#     rough_amp_fit_func = fit_generator(param_array, amp_array, rational_amp_fit)

#     # Evaluate the fit at each point in param_array
#     rough_fit_array = np.array([rough_amp_fit_func(p) for p in param_array])

#     # Convert flattened array into grid
#     rough_fit_grid = rough_fit_array.reshape(amp_grid.shape)

#     # Interpolate the quantity amp_grid/rough_fit_grid
#     interp_function = uniform_grid_interpolation_generator(amp_grid/rough_fit_grid, params_min, params_max, order=2)

#     def amp_function(params):
#         return rough_amp_fit_func(params)*interp_function(params)

#     return amp_function


def amp_function_generator(param_grid, amp_grid):
    """
        Parameters
        ----------
        param_grid : (N+1)d array
        The N Parameter values at each point on the parameter grid.
        amp_grid : Nd array
        Amplitude at each point on the parameter grid.
        
        Returns
        -------
        amp_function(params) : function
        The amplitude function that passes through the amp_grid data at the points in param_grid.
        """
    
    # Flatten param_grid and amp_grid for the fitting function
    Nparams = param_grid.shape[-1]
    params_min = param_grid[tuple([0]*Nparams)]
    params_max = param_grid[tuple([-1]*Nparams)]
    
    # Interpolate the quantity amp_grid/rough_fit_grid
    interp_function = uniform_grid_interpolation_generator(amp_grid, params_min, params_max, order=1)
    
    return interp_function


# def phase_function_generator(param_grid, phase_grid):
#     """
#     Parameters
#     ----------
#     param_grid : (N+1)d array
#         The N Parameter values at each point on the parameter grid.
#     phase_grid : Nd array
#         phase at each point on the parameter grid.

#     Returns
#     -------
#     phase_function(params) : function
#         The phase function that passes through the phase_grid data at the points in param_grid.
#     """

#     # Flatten param_grid and phase_grid for the fitting function
#     Nparams = param_grid.shape[-1]
#     params_min = param_grid[tuple([0]*Nparams)]
#     params_max = param_grid[tuple([-1]*Nparams)]

#     Npoints = product(list(param_grid.shape)[:-1])
#     param_array = param_grid.reshape((Npoints, Nparams))
#     phase_array = phase_grid.flatten()

#     # Do the fit
#     rough_phase_fit_func = fit_generator(param_array, phase_array, rational_phase_fit)

#     # Evaluate the fit at each point in param_array
#     rough_fit_array = np.array([rough_phase_fit_func(p) for p in param_array])

#     # Convert flattened array into grid
#     rough_fit_grid = rough_fit_array.reshape(phase_grid.shape)

#     # Interpolate the quantity phase_grid/rough_fit_grid
#     interp_function = uniform_grid_interpolation_generator(phase_grid/rough_fit_grid, params_min, params_max, order=2)

#     def phase_function(params):
#         return rough_phase_fit_func(params)*interp_function(params)

#     return phase_function


def phase_function_generator(param_grid, phase_grid):
    """
        Parameters
        ----------
        param_grid : (N+1)d array
        The N Parameter values at each point on the parameter grid.
        phase_grid : Nd array
        phase at each point on the parameter grid.
        
        Returns
        -------
        phase_function(params) : function
        The phase function that passes through the phase_grid data at the points in param_grid.
        """
    
    # Flatten param_grid and phase_grid for the fitting function
    Nparams = param_grid.shape[-1]
    params_min = param_grid[tuple([0]*Nparams)]
    params_max = param_grid[tuple([-1]*Nparams)]
    
    # Interpolate the quantity phase_grid/rough_fit_grid
    interp_function = uniform_grid_interpolation_generator(phase_grid, params_min, params_max, order=1)
    
    return interp_function


def amp_phase_functions_from_uniform_grid_at_empirical_nodes(param_grid, amp_at_nodes_grid, phase_at_nodes_grid):
    """
        Generate interpolating functions for amp and phase at each of the empirical_nodes T_i.
        
        Parameters
        ----------
        param_grid : (N+1)d array
        The N Parameter values at each point on the parameter grid.
        amp_at_nodes_grid : (N+1)d array
        Amplitude at the empirical_nodes at each point on the parameter grid.
        phase_at_nodes_grid : (N+1)d array
        Phase at the empirical_nodes at each point on the parameter grid.
        
        Returns
        -------
        amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
        phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.
        """
    
    # Number of empirical nodes T_i
    Nnodes = amp_at_nodes_grid.shape[-1]
    
    # Generate amp(params) function at each empirical node
    amp_function_list = []
    for nodei in range(Nnodes):
        print nodei,
        amp_grid = amp_at_nodes_grid[..., nodei]
        amp_function = amp_function_generator(param_grid, amp_grid)
        amp_function_list.append(amp_function)
    
    # Generate phase(params) function at each empirical node
    phase_function_list = []
    for nodei in range(Nnodes):
        print nodei,
        phase_grid = phase_at_nodes_grid[..., nodei]
        phase_function = phase_function_generator(param_grid, phase_grid)
        phase_function_list.append(phase_function)
    
    return amp_function_list, phase_function_list





















