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
import pycbc.types # TimeSeries
import pycbc.waveform # amplitude and phase functions

import greedy
import empiricalinterpolation as eim
import timeseriesutils as tsutils
import functionapproximation as approx


################################################################################
#                     ReducedOrderModels for TimeSeries                        #
################################################################################


################################################################################
#    Calculate the empirical nodes and interpolating TimeSeries.               #
################################################################################

def empirical_interpolation_for_time_series(waveforms):
    """Calculate empirical nodes and corresponding empirical interpolating functions.
    
    Parameters
    ----------
    waveforms : HDF5TimeSeriesSet
    
    Returns
    -------
    empirical_node_indices : List of ints
        The indices of the empirical nodes in the TimeSeries.
    B_j : List of TimeSeries
        The empirical interpolating functions
        that are 1 at the node T_j and
        0 at the other nodes T_i (for i!=j).
    """
    Nwave = len(waveforms.get_parameters())
    
    # Convert the list of TimeSeries to a list of numpy arrays
    #wave_np = [waveforms[i].numpy() for i in range(Nwave)]
    wave_np = [waveforms.get_waveform_data(i).numpy() for i in range(Nwave)]
    
    # Determine the empirical nodes
    empirical_node_indices = eim.generate_empirical_nodes(wave_np)
    
    # Determine the empirical interpolating functions B_j(t)
    B_j_np = eim.generate_interpolant_list(wave_np, empirical_node_indices)
    
    # Convert the arrays to TimeSeries.
    delta_t = waveforms.get_waveform_data(0).delta_t
    epoch = waveforms.get_waveform_data(0).start_time
    B_j = [pycbc.types.TimeSeries(B_j_np[j], delta_t=delta_t, epoch=epoch) for j in range(Nwave)]
    
    return empirical_node_indices, B_j


#def empirical_interpolation_for_time_series(waveforms):
#    """Calculate empirical nodes and corresponding empirical interpolating functions.
#    
#    Parameters
#    ----------
#    waveforms : List of TimeSeries
#    
#    Returns
#    -------
#    empirical_node_indices : List of ints
#        The indices of the empirical nodes in the TimeSeries.
#    B_j : List of TimeSeries
#        The empirical interpolating functions
#        that are 1 at the node T_j and
#        0 at the other nodes T_i (for i!=j).
#    """
#    Nwave = len(waveforms)
#    
#    # Convert the list of TimeSeries to a list of numpy arrays
#    wave_np = [waveforms[i].numpy() for i in range(Nwave)]
#    
#    # Determine the empirical nodes
#    empirical_node_indices = eim.generate_empirical_nodes(wave_np)
#    
#    # Determine the empirical interpolating functions B_j(t)
#    B_j_np = eim.generate_interpolant_list(wave_np, empirical_node_indices)
#    
#    # Convert the arrays to TimeSeries.
#    delta_t = waveforms[0].delta_t
#    epoch = waveforms[0].start_time
#    B_j = [pycbc.types.TimeSeries(B_j_np[j], delta_t=delta_t, epoch=epoch) for j in range(Nwave)]
#    
#    return empirical_node_indices, B_j




################################################################################
#  Get the amplitude and phase of the TimeSeries at the empirical nodes T_j.   #
################################################################################

def amp_phase_at_empirical_nodes(waveforms, empirical_node_indices):
    """Determine the coefficients of the amplitude and phase fits at the empirical_nodes T_j.
    
    Parameters
    ----------
    waveforms : HDF5TimeSeriesSet
    empirical_node_indices : List of integers
        Indices corresponding to the nodes [T_0, ..., T_{Nnodes-1}].
    
    Returns
    -------
    amp_at_nodes : 2d array
        Amplitude at the empirical_nodes with each waveform in a row.
    phase_at_nodes : 2d array
        Phase at the empirical_nodes with each waveform in a row.
    """
    # Number of waveforms
    Nwave = len(waveforms.get_parameters())
    
    print 'Getting the amplitude at the list of empirical nodes T_j for each waveform i...'
    amp_at_nodes = np.array([
            waveforms.get_waveform_data(i, data_set='amplitude')[empirical_node_indices] 
            for i in range(Nwave)])
    
    print 'Getting the phase at the list of empirical nodes T_j for each waveform i...'
    phase_at_nodes = np.array([
            waveforms.get_waveform_data(i, data_set='phase')[empirical_node_indices] 
            for i in range(Nwave)])

    return amp_at_nodes, phase_at_nodes


#def amp_phase_at_empirical_nodes(amplist, phaselist, empirical_node_indices):
#    """Determine the coefficients of the amplitude and phase fits at the empirical_nodes T_i.
#    
#    Parameters
#    ----------
#    amp_list : List of arraylike objects
#        Amplitude of waveforms.
#    phase_list : List of arraylike objects
#        Phase of waveforms.
#    empirical_node_indices : List of integers
#        Indices corresponding to the nodes [T_0, ..., T_{j-1}].
#    
#    Returns
#    -------
#    amp_at_nodes : 2d array
#        Amplitude at the empirical_nodes with each waveform waveform in a row.
#    phase_at_nodes : 2d array
#        Phase at the empirical_nodes with each waveform waveform in a row.
#    """
#    # Number of waveforms
#    Nwave = len(amplist)
#    # Dimension of reduced basis
#    m = len(empirical_node_indices)
#    
#    # Get amplitude and phase at each empirical node T_i
#    amp_at_nodes = np.array([[amplist[i][empirical_node_indices[j]] for j in range(m)] for i in range(Nwave)])
#    phase_at_nodes = np.array([[phaselist[i][empirical_node_indices[j]] for j in range(m)] for i in range(Nwave)])
#    
#    return amp_at_nodes, phase_at_nodes


def generate_rom_from_waveforms_on_regular_grid(ts, rb):
    """
    Generate a reduced order model by interpolating 
    the amplitude and phase of the training set
    which is on a grid of the waveform parameters.
    
    Parameters
    ----------
    ts : HDF5TimeSeriesSet
        The training set with a regular_grid_shape.
    rb : HDF5TimeSeriesSet
        The reduced basis. It's not on a regular grid.
    
    Returns
    -------
    rom : ReducedOrderModelTimeSeries
        The reduced order model object.
    train_amp_grid : (N+1)d array
        Amplitude at the empirical_nodes at each point on the training-set parameter grid.
    train_phase_grid : (N+1)d array
        Phase at the empirical_nodes at each point on the training-set parameter grid.
    empirical_node_indices : List of integers
        Indices corresponding to the nodes [T_0, ..., T_{j-1}].
    B_j: List of 1d arrays
        The interpolating functions.
    """
    print 'Calculating empirical nodes and empirical interpolating functions...'
    empirical_node_indices, B_j = empirical_interpolation_for_time_series(rb)
    
    # Calculate the amplitude and phase at nodes for every waveform in training set
    train_amp_at_nodes, train_phase_at_nodes = amp_phase_at_empirical_nodes(ts, empirical_node_indices)
    
    # Turn parameters list into a grid
    Nnodes = len(train_amp_at_nodes[0])
    param_grid = ts.get_parameter_grid()
    
    # Reshape the amp_at_nodes and phase_at_nodes 2d-arrays into (N+1)d arrays
    shape = tuple(ts.regular_grid_shape+[Nnodes])
    train_amp_grid = train_amp_at_nodes.reshape(shape)
    train_phase_grid = train_phase_at_nodes.reshape(shape)
    
    print 'Generating amp_function_list and phase_function_list using interpolation...'
    amp_function_list, phase_function_list = approx.amp_phase_functions_from_uniform_grid_at_empirical_nodes(
        param_grid, train_amp_grid, train_phase_grid)
    
    print 'Constructing the ROM object...'
    rom = ReducedOrderModelTimeSeries(B_j, amp_function_list, phase_function_list)
    
    return rom, train_amp_grid, train_phase_grid, empirical_node_indices, B_j


################################################################################
#        Functions and class for calculating the ROM TimeSeries                #
################################################################################

def reduced_order_model_time_series_waveform(params, B_j, amp_function_list, phase_function_list):
    """Calculate the reduced order model waveform. This is the online stage.
    
    Parameters
    ----------
    param : 1d array
        Physical waveform parameters.
    B_j : List of TimeSeries
        List of the interpolants.
    amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
    phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.
    
    Returns
    -------
    hinterp : TimeSeries
        Reduced order model waveform
    """
    Nnodes = len(amp_function_list)
    
    # Calculate waveform at nodes
    waveform_at_nodes = np.array([amp_function_list[j](params)*np.exp(1.0j*phase_function_list[j](params)) for j in range(Nnodes)])
    
    B_j_array = np.array([B_j[j].numpy() for j in range(len(B_j))])
    hinterp = np.dot(waveform_at_nodes, B_j_array)
    
    delta_t = B_j[0].delta_t
    epoch = B_j[0].start_time
    return pycbc.types.TimeSeries(hinterp, delta_t=delta_t, epoch=epoch)


class ReducedOrderModelTimeSeries:
    """Construct a reduced order model for a TimeSeries waveform.
    
    Attributes
    ----------
    B_j
    amp_function_list
    phase_function_list
    """
    
    def __init__(self, B_j, amp_function_list, phase_function_list):
        """Initialize the reduced order model. Takes precomputed interpolating
        TimeSeries, and fuctions that interpolate the amplitude and phase at each
        empirical node as a function of the waveform parameters.
        
        Parameters
        ----------
        B_j : List of TimeSeries
            The empirical interpolating functions.
        amp_function_list : List of func(params)
            Functions for the amplitude at the empirical_nodes.
        phase_function_list : List of func(params)
            Functions for the phase at the empirical_nodes.
        """
        self.B_j = B_j
        self.amp_function_list = amp_function_list
        self.phase_function_list = phase_function_list
        
        # Check input consistency
        if any([len(B) != len(B_j[0]) for B in B_j]):
            raise Exception, 'All the interpolating functions B_j in the reduced basis must have the same length.'
        if len(self.B_j) != len(self.amp_function_list) != len(self.phase_function_list):
            raise Exception, 'B_j, amp_function_list, and phase_function_list are not the same length.'

    def evaluate(self, params):
        """Evaluate the reduced order model in dimensionless units.
        
        Parameters
        ----------
        params : array of floats
            The physical parameters of the ROM.
            
        Returns
        -------
        waveform : TimeSeries
            The ROM waveform.
        """
        return reduced_order_model_time_series_waveform(params, self.B_j, self.amp_function_list, self.phase_function_list)
    
    def evaluate_physical_units(self, phys_params):
        """Evaluate the waveform with physics time and strain units, including 
        inclination, f_low, etc.
        """
        # TODO: make this function
        pass


#class ReducedOrderModelTimeSeries(tsutils.TimeSeriesSet):
#    
#    def __init__(self, filename=None, waveforms=None, parameters=None):
#        """
#        Call the same method as TimeSeriesSet. ReducedBasis doesn't have a regular grid though.
#        """
#
#        tsutils.TimeSeriesSet.__init__(self, filename=filename, waveforms=waveforms, parameters=parameters, regular_grid_shape=None)
#        self.B_j = None
#        self.amp_function_list = None
#        self.phase_function_list = None
#        self.empirical_node_indices = None
#        
#        # !!!!!!!!! put this and many other checks in a check_consistency function in the TimeSeriesSet class
#        # Check that all the reduced basis waveforms have the same length
#        if any([len(w) != len(waveforms[0]) for w in waveforms]):
#            raise Exception, 'All the waveforms in the reduced basis must have the same length.'
#
##    def generate_empirical_nodes(self):
##        """
##        Find the location of the empirical nodes.
##        """
##        
##        # Convert the RB TimeSeries to numpy arrays
##        rb_arrays = [self.waveforms[i].numpy() for i in range(len(self.waveforms))]
##        
##        # Calculate the indices for the empirical nodes
##        self.empirical_node_indices = eim.generate_empirical_nodes(rb_arrays)
##    
##    def generate_interpolant_time_series_list(self):
##        """
##        Get the interpolating functions B_j(t).
##        """
##        
##        # Calculate a list of arrays corresponding to B_j(t)
##        B_j_arrays = eim.generate_interpolant_list(self.waveforms, self.empirical_node_indices)
##        
##        # Convert the arrays to TimeSeries.
##        delta_t = self.waveforms[0].delta_t
##        epoch = self.waveforms[0].start_time
##        m = len(B_j_arrays)
##        self.B_j = [pycbc.types.TimeSeries(B_j_arrays[j], delta_t=delta_t, epoch=epoch) for j in range(m)]
#
#    def evaluate(self, params):
#        """
#        Evaluate the reduced order model at the parameter values params.
#        """
#
#        # Check that B_j, amp_function_list, phase_function_list have been set and are the same length
#        if self.B_j is None: raise Exception, 'self.B_j is not provided.'
#        if self.amp_function_list is None: raise Exception, 'self.amp_function_list is not provided.'
#        if self.phase_function_list is None: raise Exception, 'self.phase_function_list is not provided.'
#        if len(self.B_j) == len(self.amp_function_list) == len(self.amp_function_list):
#            # Actually evaluate the ROM waveform
#            return reduced_order_model_time_series_waveform(params, self.B_j, self.amp_function_list, self.phase_function_list)
#        else:
#            raise Exception, 'self.B_j, self.amp_function_list, and self.phase_function_list are not the same length.'
#
