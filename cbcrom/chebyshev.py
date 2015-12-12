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

"""
Utilities for Chebyshev interpolation.
"""

import numpy as np
import scipy.fftpack
import copy
import h5py

################################################################################
# Functions for Chebyshev Polynomials and finding the nodes
################################################################################

def chebyshev_polynomial(n, x):
    """n-th order Chebyshev polynomial T_n(x) of the first kind.
    
    Parameters
    ----------
    n : int
        Order of the polynomial >= 0.
    x : float 
        Must by evaluated in the interval [-1, 1].
    """
    return np.cos(n*np.arccos(x))


def chebyshev_lobatto_nodes(Nnodes, low=-1.0, high=1.0):
    """Find the Chebyshev-Lobatto nodes in the interval [low, high].
    
    Parameters
    ----------
    Nnodes : int
        Number of nodes
    low : {float, -1.0} 
        Lower bound of the interval.
    high : {float, 1.0}
        Upper bound of the invterval
    """
    Norder=Nnodes-1
    k = np.arange(Norder+1)
    x_k = -np.cos(k*np.pi/Norder)
    y_k = 0.5*(high-low)*x_k + 0.5*(low+high)
    return y_k





################################################################################
# Fast Chebyshev transform in 1D, 2D, 3D
################################################################################

def fast_chebyshev_transform(f_k):
    """
    Compute the Chebyshev transform of f_k.
    
    Paramaters
    ----------
    f_k : real numpy array
        Function evaluated at the Chebyshev-Gauss-Lobatto nodes in interval [-1, 1].
    
    Returns
    -------
    b_n : real numpy array
        Coefficients of the Chebyshev polynomials.
    """
    Npoints = len(f_k)
    
    # Make array of (-1)**n = [1, -1, 1, -1, 1, ...]
    sign = np.empty(Npoints)
    sign[::2] = 1
    sign[1::2] = -1
    
    # Do type 1 cosine transform (which multiplies all points except first and last by 2)
    b_n = scipy.fftpack.dct(f_k, type=1, n=None, axis=-1, norm=None, overwrite_x=False)/(Npoints-1.0)
    
    # Flip the sign of every other coefficient and divide first and last coefficient by 2
    b_n = np.multiply(b_n, sign)
    b_n[0] *= 0.5
    b_n[-1] *= 0.5
    
    return b_n


def fast_chebyshev_transform_2d(f_ij):
    """Calculate the coefficients c_lm of the Chebyshev polynomial product T_l(x)T_m(y)
    in the Chebyshev expansion of the function f(x, y).
    
    Parameters
    ----------
    f_ij : 2d array of floats
        Value of the function f(x_i, y_j) at the Chebyshev-Lobatto nodes x_i, y_j.
    
    Returns
    -------
    c_lmn : 2d array of float
        Coefficients of T_l(x)T_m(y).
    """
    # Number of points in each dimension
    Nx, Ny = f_ij.shape
    
    # Make a copy of f_ij instead of just referencing it with = 
    # so you don't overwrite f_ij
    c_lm = copy.copy(f_ij)
    
    # Transform each column of the grid
    # Then repeat for each parameter
    for j in range(Ny):
        c_lm[:, j] = fast_chebyshev_transform(c_lm[:, j])
      
    for i in range(Nx):
            c_lm[i, :] = fast_chebyshev_transform(c_lm[i, :])
    
    return c_lm


def fast_chebyshev_transform_3d(f_ijk):
    """Calculate the coefficients c_lmn of the Chebyshev polynomial product T_l(x)T_m(y)T_n(z)
    in the Chebyshev expansion of the function f(x, y, z).
    
    Parameters
    ----------
    f_ijk : 3d array of floats
        Value of the function f(x_i, y_j, z_k) at the Chebyshev-Lobatto nodes x_i, y_j, z_k.
    
    Returns
    -------
    c_lmn : 3d array of float
        Coefficients of T_l(x)T_m(y)T_n(z).
    """
    # Number of points in each dimension
    Nx, Ny, Nz = f_ijk.shape
    
    # Make a copy of f_ijk instead of just referencing it with = 
    # so you don't overwrite f_ijk
    c_lmn = copy.copy(f_ijk)
    
    # Transform each column of the grid
    # Then repeat for each parameter
    for j in range(Ny):
        for k in range(Nz): 
            c_lmn[:, j, k] = fast_chebyshev_transform(c_lmn[:, j, k])
    
    for k in range(Nz):
        for i in range(Nx): 
            c_lmn[i, :, k] = fast_chebyshev_transform(c_lmn[i, :, k])
    
    for i in range(Nx):
        for j in range(Ny): 
            c_lmn[i, j, :] = fast_chebyshev_transform(c_lmn[i, j, :])
    
    return c_lmn


################################################################################
# Perform the interpolation in 1D, 2D, 3D
################################################################################

def chebyshev_interpolation(x, c_l_array, xlow, xhigh):
    """Interpolate f(x) given its expansion in terms of Chebyshev polynomials T_l(x).
        
        Parameters
        ----------
        x : float
        c_l_array : 1d array
        Coefficients of the Chebyshev polynomials
        xlow : float
        Lower bound of the interval.
        xhigh : float
        Upper bound of the interval.
        
        Returns
        -------
        f(x) : float
        Function evaluated at x.
        """
    xrescale = (x-0.5*(xhigh+xlow)) / (0.5*(xhigh-xlow))
    return np.polynomial.chebyshev.chebval(xrescale, c_l_array)


def chebyshev_interpolation2d(x, y, c_lm_array, xlow, xhigh, ylow, yhigh):
    """Interpolate f(x, y) given its expansion in terms of Chebyshev polynomials T_l(x)T_m(y).
        
        Parameters
        ----------
        x, y : floats
        c_lm_array : 2d array
        Coefficients of the Chebyshev polynomials
        xlow, ylow : floats
        Lower bounds of the intervals.
        xhigh, yhigh : floats
        Upper bounds of the intervals.
        
        Returns
        -------
        f(x, y) : float
        Function evaluated at x, y.
        """
    xrescale = (x-0.5*(xhigh+xlow)) / (0.5*(xhigh-xlow))
    yrescale = (y-0.5*(yhigh+ylow)) / (0.5*(yhigh-ylow))
    return np.polynomial.chebyshev.chebval2d(xrescale, yrescale, c_lm_array)


def chebyshev_interpolation3d(x, y, z, c_lmn_array, xlow, xhigh, ylow, yhigh, zlow, zhigh):
    """Interpolate f(x, y, z) given its expansion in terms of Chebyshev polynomials T_l(x)T_m(y)T_n(z).
        
        Parameters
        ----------
        x, y, z : floats
        c_lmn_array : 3d array
        Coefficients of the Chebyshev polynomials
        xlow, ylow, zlow : floats
        Lower bounds of the intervals.
        xhigh, yhigh, zhigh : floats
        Upper bounds of the intervals.
        
        Returns
        -------
        f(x, y, z) : float
        Function evaluated at x, y, z.
        """
    xrescale = (x-0.5*(xhigh+xlow)) / (0.5*(xhigh-xlow))
    yrescale = (y-0.5*(yhigh+ylow)) / (0.5*(yhigh-ylow))
    zrescale = (z-0.5*(zhigh+zlow)) / (0.5*(zhigh-zlow))
    return np.polynomial.chebyshev.chebval3d(xrescale, yrescale, zrescale, c_lmn_array)


################################################################################
# Class for approximating 1D, 2D, 3D functions with Chebyshev polynomials
################################################################################

class ChebyshevApproximation:
    """Methods for approximating [1, 2, 3]-d functions with Chebyshev polynomials.
    
    Attributes
    ----------
    shape : List of length Ndim
        Number of points in each dimension
    low : List of length Ndim
        Lower boundary of grid
    high : List of length Ndim
        Upper boundary of grid
    Ndim : int
        Number of dimensions
    nodes : Ndim length list of arrays containing the nodes in each dimension.
        Coordinates of the Lobatto nodes
    f_grid : Array of shape shape.
        Function values at the grid of Lobatto nodes.
    coefficients : Array of shape shape.
        Coefficients of the Chebyshev expansion.
    """
    
    def __init__(self):
        self.shape = None
        self.low = None
        self.high = None
        self.Ndim = None
        self.nodes = None
        self.f_grid = None
        self.coefficients = None
        
    def lobatto_nodes(self, shape, low, high):
        """Generate a list of arrays for the Lobatto nodes in each dimension.
        """
        self.shape = shape
        self.low = low
        self.high = high
        self.Ndim = len(shape)
        
        if self.Ndim not in [1, 2, 3]:
            raise Exception, 'Only 1, 2, and 3 dimensions are currently supported.'
            
        self.nodes = []
        for i in range(self.Ndim):
            nodes_i = chebyshev_lobatto_nodes(shape[i], low=low[i], high=high[i])
            self.nodes.append(nodes_i)
    
    def evaluate_function_at_lobatto_nodes(self, func, shape, low, high):
        """Evaluate a function on a grid given by the Lobatto nodes.
        """
        self.lobatto_nodes(shape, low, high)
        
        if self.Ndim == 1:
            x_i = self.nodes[0]
            self.f_grid = np.array([func(x) for x in x_i])
        elif self.Ndim == 2:
            x_i, y_j = self.nodes
            self.f_grid = np.array([[func(x, y) for y in y_j] for x in x_i])
        elif self.Ndim == 3:
            x_i, y_j, z_k = self.nodes
            self.f_grid = np.array([[[func(x, y, z) for z in z_k] for y in y_j] for x in x_i])
        else:
            raise Exception, 'Only 1, 2, and 3 dimensions are currently supported.'
        
    def calculate_chebyshev_coefficients(self):
        """Calculate the coefficients in the Chebyshev expansion.
        """
        if self.Ndim == 1:
            self.coefficients = fast_chebyshev_transform(self.f_grid)
        elif self.Ndim == 2:
            self.coefficients = fast_chebyshev_transform_2d(self.f_grid)
        elif self.Ndim == 3:
            self.coefficients = fast_chebyshev_transform_3d(self.f_grid)
        else:
            raise Exception, 'Only 1, 2, and 3 dimensions are currently supported.'
            
    def interpolation(self, *args):
        """Interpolate the function at a specific point.
        """
        if self.Ndim == 1:
            # args is a tuple
            x = args[0]
            coeff = self.coefficients
            xlow = self.low[0]
            xhigh = self.high[0]
            return chebyshev_interpolation(x, coeff, xlow, xhigh)
        if self.Ndim == 2:
            x, y = args
            coeff = self.coefficients
            xlow, ylow = self.low
            xhigh, yhigh = self.high
            return chebyshev_interpolation2d(x, y, coeff, xlow, xhigh, ylow, yhigh)
        if self.Ndim == 3:
            x, y, z = args
            coeff = self.coefficients
            xlow, ylow, zlow = self.low
            xhigh, yhigh, zhigh = self.high
            return chebyshev_interpolation3d(x, y, z, coeff, xlow, xhigh, ylow, yhigh, zlow, zhigh)


################################################################################
#      Generating lists of interpolating functions and saving them
################################################################################


def chebyshev_interpolation3d_generator(coefficients, params_min, params_max):
    """Generate a function of N parameters that interpolates a uniformly spaced Nd array.
    
    Parameters
    ----------
    coefficients : N-d numpy array
        The chebyshev coefficients corresponding to a list of functions.
    params_min : 1d array of length 3
        Minimum value of each parameter.
    params_max : 1d array of length 3
        Maximum value of each parameter.
    
    Returns
    -------
    function_list : list of function(params)
    """
    cheb_approx = ChebyshevApproximation()
    cheb_approx.low = params_min
    cheb_approx.high = params_max
    cheb_approx.Ndim = 3
    cheb_approx.coefficients = coefficients
    def interp(params):
        return cheb_approx.interpolation(params[0], params[1], params[2])
    
    return interp


def chebyshev_coefficient3d_list_generator(f_ijk_list, params_min, params_max):
    """
    Generate Chebyshev coefficients to approximate the function (amp or phase) at the empirical_nodes T_i.
    
    Parameters
    ----------
    f_ijk_list : list of N-d arrays
        Function values on the Chebyshev-Gauss-Lobatto grid.
    params_min : 1d array of length 3
        Minimum value of each parameter.
    params_max : 1d array of length 3
        Maximum value of each parameter.
    
    Returns
    -------
    coefficient_list : List of N-d numpy arrays
        The chebyshev coefficients corresponding to a list of functions.
    """
    
    # Number of empirical nodes T_i
    Nnodes = len(f_ijk_list)
    
    # Generate amp(params) function at each empirical node
    coefficient_list = []
    for i in range(Nnodes):
        cheb_approx = ChebyshevApproximation()
        # Get the bounds of the parameter space
        cheb_approx.low = params_min
        cheb_approx.high = params_max
        cheb_approx.Ndim = 3
        cheb_approx.f_grid = f_ijk_list[i]
        # Calculate the Chebyshev coefficients
        cheb_approx.calculate_chebyshev_coefficients()
        coefficient_list.append(cheb_approx.coefficients)
    
    return coefficient_list


def save_chebyshev_coefficients_list(filename, coefficients_list, params_min, params_max):
    """
    Load a list of N-d coefficient arrays.
    
    Parameters
    ----------
    filename : string
        hdf5 filename
    coefficients_list : List of N-d numpy arrays
        The chebyshev coefficients corresponding to a list of functions.
    params_min : 1d array of length N
        Minimum value of each parameter.
    params_max : 1d array of length N
        Maximum value of each parameter.
    """
    coeff_file = h5py.File(filename)
    
    Ncoeff = len(coefficients_list)
    for i in range(Ncoeff):
        coeff_file['coeff_'+str(i)] = coefficients_list[i]
    
    coeff_file['params_min'] = params_min
    coeff_file['params_max'] = params_max
    coeff_file.close()


def load_chebyshev_coefficients_list(filename):
    """
    Load a list of N-d coefficient arrays.
    
    Parameters
    ----------
    filename : string
        hdf5 filename
    
    Returns
    -------
    coefficients_list : List of N-d numpy arrays
        The chebyshev coefficients corresponding to a list of functions.
    params_min : 1d array of length N
        Minimum value of each parameter.
    params_max : 1d array of length N
        Maximum value of each parameter.
    """
    coeff_file = h5py.File(filename)
    
    # Get number of coefficients arrays
    names = coeff_file.keys()
    Ncoeffs = len([names[i] for i in range(len(names)) if 'coeff_' in names[i]])
    
    # Extract the coefficients arrays
    coefficients_list = [coeff_file['coeff_'+str(i)][:] for i in range(Ncoeffs)]
    
    # Get parameter range
    params_min = coeff_file['params_min'][:]
    params_max = coeff_file['params_max'][:]
    
    coeff_file.close()
    
    return coefficients_list, params_min, params_max




