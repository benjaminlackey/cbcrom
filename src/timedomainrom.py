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
Methods for constructing ROM from TimeDomainWaveform reduced basis.
"""

import numpy as np
import scipy.interpolate
#import scipy.integrate

#import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter
#from matplotlib.pylab import subplots_adjust
#
#import copy
#import h5py
#
## TimeSeries
#import pycbc.types

import empiricalinterpolation as eim
import timedomainwaveform as tdwave
import chebyshev as cheb


################################################################################
#              Empirical interpolation for TimeDomainWaveforms                 #
################################################################################

#def empirical_interpolation_for_time_domain_waveform(waveforms):
#    """Calculate empirical nodes and corresponding empirical interpolating functions.
#
#    Parameters
#    ----------
#    waveforms : HDF5TimeDomainWaveformSet
#
#    Returns
#    -------
#    empirical_node_indices : List of ints
#        The indices of the empirical nodes in the TimeDomainWaveforms.
#    B_j : List of TimeDomainWaveform
#        The empirical interpolating functions
#        that are 1 at the node T_j and
#        0 at the other nodes T_i (for i!=j).
#    """
#    Nwave = waveforms.get_len()
#
#    # Convert the list of TimeDomainWaveform to a list of complex numpy arrays
#    wave_np = [waveforms.get_waveform_data(i).get_complex() for i in range(Nwave)]
#
#    # Determine the empirical nodes
#    empirical_node_indices = eim.generate_empirical_nodes(wave_np)
#
#    # Determine the empirical interpolating functions B_j(t)
#    B_j_np = eim.generate_interpolant_list(wave_np, empirical_node_indices)
#
#    # Convert the arrays to TimeDomainWaveforms.
#    time = waveforms.get_waveform_data(0).time
#    B_j = []
#    for j in range(Nwave):
#        comp = B_j_np[j]
#        amp = np.abs(comp)
#        phase = np.angle(comp)
#        phase = np.unwrap(phase)
#        B_j.append(TimeDomainWaveform(time, amp, phase))
#
#    return empirical_node_indices, B_j

def empirical_interpolation_for_time_domain_waveform(waveforms, datatype):
    """Calculate empirical nodes and corresponding empirical interpolating functions
    from a set of reduced basis waveforms.
    
    Parameters
    ----------
    waveforms : HDF5TimeDomainWaveformSet
    datatype : string {'complex', 'amp', 'phase'}
    
    Returns
    -------
    empirical_node_indices : List of ints
        The indices of the empirical nodes in the TimeDomainWaveforms.
    B_j : List of TimeDomainWaveform
        The empirical interpolating functions
        that are 1 at the node T_j and
        0 at the other nodes T_i (for i!=j).
    """
    Nwave = waveforms.get_len()
    
    # Convert the list of TimeDomainWaveform to a list of complex numpy arrays
    if datatype == 'complex':
        wave_np = [waveforms.get_waveform_data(i).get_complex() for i in range(Nwave)]
    elif datatype == 'amp':
        wave_np = [waveforms.get_waveform_data(i).amp for i in range(Nwave)]
    elif datatype == 'phase':
        wave_np = [waveforms.get_waveform_data(i).phase for i in range(Nwave)]
    else:
        raise Exception, "datatype must be one of {'complex', 'amp', 'phase'}."
    
    # Determine the empirical nodes
    empirical_node_indices = eim.generate_empirical_nodes(wave_np)
    
    # Determine the empirical interpolating functions B_j(t)
    B_j_np = eim.generate_interpolant_list(wave_np, empirical_node_indices)
    
    # Convert the arrays to TimeDomainWaveforms.
    time = waveforms.get_waveform_data(0).time
    B_j = []
    for j in range(Nwave):
        if datatype == 'complex':
            comp = B_j_np[j]
            B_j.append(complex_to_time_domain_waveform(time, comp))
        elif datatype == 'amp':
            amp = B_j_np[j]
            B_j.append(tdwave.TimeDomainWaveform(time, amp, np.zeros(len(time))))
        elif datatype == 'phase':
            phase = B_j_np[j]
            B_j.append(tdwave.TimeDomainWaveform(time, np.zeros(len(time)), phase))
        else:
            raise Exception, "datatype must be one of {'complex', 'amp', 'phase'}."
    
    return empirical_node_indices, B_j


#def amp_phase_at_empirical_nodes(waveforms, empirical_node_indices):
#    """Determine the coefficients of the amplitude and phase fits at the empirical_nodes T_j.
#
#    Parameters
#    ----------
#    waveforms : HDF5TimeDomainWaveformSet
#    empirical_node_indices : List of integers
#        Indices corresponding to the nodes [T_0, ..., T_{Nnodes-1}].
#
#    Returns
#    -------
#    amp_at_nodes : 2d array
#        Amplitude at the empirical_nodes with each waveform in a row.
#    phase_at_nodes : 2d array
#        Phase at the empirical_nodes with each waveform in a row.
#    """
#    # Number of waveforms
#    Nwave = waveforms.get_len()
#
#    print 'Getting the amplitude at the list of empirical nodes T_j for each waveform i...'
#    amp_at_nodes = np.array([
#            waveforms.get_waveform_data(i).amp[empirical_node_indices]
#            for i in range(Nwave)])
#
#    print 'Getting the phase at the list of empirical nodes T_j for each waveform i...'
#    phase_at_nodes = np.array([
#            waveforms.get_waveform_data(i).phase[empirical_node_indices]
#            for i in range(Nwave)])
#
#    return amp_at_nodes, phase_at_nodes


def waveform_data_at_empirical_nodes(waveforms, empirical_node_indices, datatype):
    """Determine the coefficients of the amplitude and phase fits at the empirical_nodes T_j.
    
    Parameters
    ----------
    waveforms : HDF5TimeDomainWaveformSet
    empirical_node_indices : List of integers
        Indices corresponding to the nodes [T_0, ..., T_{Nnodes-1}].
    datatype : string {'amp', 'phase'}
    
    Returns
    -------
    data_at_nodes : 2d array
        Waveform data at the empirical_nodes with each waveform in a row.
    """
    # Number of waveforms
    Nwave = waveforms.get_len()
    
    if datatype == 'amp':
        data_at_nodes = np.array([
                                  waveforms.get_waveform_data(i).amp[empirical_node_indices]
                                  for i in range(Nwave)])
    elif datatype == 'phase':
        data_at_nodes = np.array([
                                  waveforms.get_waveform_data(i).phase[empirical_node_indices]
                                  for i in range(Nwave)])
    
    return data_at_nodes


################################################################################
#                 ReducedOrderModelTimeDomainWaveform class                    #
################################################################################

#def reduced_order_model_time_domain_waveform(params, B_j, amp_function_list, phase_function_list):
#    """Calculate the reduced order model waveform. This is the online stage.
#    
#    Parameters
#    ----------
#        param : 1d array
#        Physical waveform parameters.
#        B_j : List of TimeDomainWaveform
#        List of the interpolants.
#        amp_function_list : List of interpolating functions
#        List of interpolating functions for the amplitude at the empirical_nodes.
#        phase_function_list : List of arrays
#        List of interpolating functions for the phase at the empirical_nodes.
#        
#        Returns
#        -------
#        hinterp : TimeDomainWaveform
#        Reduced order model waveform
#        """
#    Nnodes = len(amp_function_list)
#    
#    # Calculate waveform at nodes
#    amp_at_nodes = np.array([amp_function_list[j](params) for j in range(Nnodes)])
#    phase_at_nodes = np.array([phase_function_list[j](params) for j in range(Nnodes)])
#    h_at_nodes = amp_at_nodes*np.exp(1.0j*phase_at_nodes)
#    
#    # Get complex version of B_j's in array form
#    B_j_array = np.array([B_j[j].get_complex() for j in range(len(B_j))])
#    
#    # Evaluate waveform
#    hinterp = np.dot(h_at_nodes, B_j_array)
#    
#    # Rewrite as TimeDomainWaveform
#    hinterp_time = B_j[0].time
#    hinterp_amp = np.abs(hinterp)
#    hinterp_phase = np.angle(hinterp)
#    hinterp_phase = np.unwrap(hinterp_phase)
#    return TimeDomainWaveform(hinterp_time, hinterp_amp, hinterp_phase)


def reduced_order_model_time_domain_waveform(params, Bamp_j, Bphase_j, amp_function_list, phase_function_list):
    """Calculate the reduced order model waveform. This is the online stage.
    
    Parameters
    ----------
    param : 1d array
        Physical waveform parameters.
    Bamp_j : List of TimeDomainWaveform
        List of the ampltude interpolants.
    Bphase_j : List of TimeDomainWaveform
        List of the phase interpolants.
    amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
    phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.
    
    Returns
    -------
    hinterp : TimeDomainWaveform
        Reduced order model waveform
    """
    Namp_nodes = len(amp_function_list)
    Nphase_nodes = len(phase_function_list)
    
    # Calculate waveform at nodes
    amp_at_nodes = np.array([amp_function_list[j](params) for j in range(Namp_nodes)])
    phase_at_nodes = np.array([phase_function_list[j](params) for j in range(Nphase_nodes)])
    
    # Get complex version of B_j's in array form
    Bamp_j_array = np.array([Bamp_j[j].amp for j in range(Namp_nodes)])
    Bphase_j_array = np.array([Bphase_j[j].phase for j in range(Nphase_nodes)])
    
    # Evaluate waveform
    amp_interp = np.dot(amp_at_nodes, Bamp_j_array)
    phase_interp = np.dot(phase_at_nodes, Bphase_j_array)
    
    # Rewrite as TimeDomainWaveform 
    time = Bamp_j[0].time
    return tdwave.TimeDomainWaveform(time, amp_interp, phase_interp)


#class ReducedOrderModelTimeDomainWaveform:
#    """Construct a reduced order model for a TimeSeries waveform.
#        
#        Attributes
#        ----------
#        B_j
#        amp_function_list
#        phase_function_list
#        """
#    
#    def __init__(self, B_j, amp_function_list, phase_function_list):
#        """Initialize the reduced order model. Takes precomputed interpolating
#            TimeSeries, and fuctions that interpolate the amplitude and phase at each
#            empirical node as a function of the waveform parameters.
#            
#            Parameters
#            ----------
#            B_j : List of TimeDomainWaveform
#            The empirical interpolating functions.
#            amp_function_list : List of func(params)
#            Functions for the amplitude at the empirical_nodes.
#            phase_function_list : List of func(params)
#            Functions for the phase at the empirical_nodes.
#            """
#        self.B_j = B_j
#        self.amp_function_list = amp_function_list
#        self.phase_function_list = phase_function_list
#    # TODO: Check for correct input.
#    
#    def evaluate(self, params):
#        """Evaluate the reduced order model in dimensionless units.
#            
#            Parameters
#            ----------
#            params : array of floats
#            The physical parameters of the ROM.
#            
#            Returns
#            -------
#            waveform : TimeDomainWaveform
#            The ROM waveform.
#            """
#        return reduced_order_model_time_domain_waveform(params, self.B_j, self.amp_function_list, self.phase_function_list)
#    
#    def evaluate_physical_units(self, phys_params):
#        """Evaluate the waveform with physics time and strain units, including 
#            inclination, f_low, etc.
#            """
#        # TODO: make this function
#        pass


class ReducedOrderModelTimeDomainWaveform:
    """Construct a reduced order model for a TimeSeries waveform.
    
    Attributes
    ----------
    Bamp_j
    Bphase_j
    amp_function_list
    phase_function_list
    """
    
    def __init__(self, Bamp_j, Bphase_j, amp_function_list, phase_function_list):
        """Initialize the reduced order model. Takes precomputed interpolating
        TimeSeries, and fuctions that interpolate the amplitude and phase at each
        empirical node as a function of the waveform parameters.
        
        Parameters
        ----------
        param : 1d array
            Physical waveform parameters.
        Bamp_j : List of TimeDomainWaveform
            List of the ampltude interpolants.
        Bphase_j : List of TimeDomainWaveform
            List of the phase interpolants.
        amp_function_list : List of interpolating functions
            List of interpolating functions for the amplitude at the empirical_nodes.
        phase_function_list : List of arrays
            List of interpolating functions for the phase at the empirical_nodes.
        """
        self.Bamp_j = Bamp_j
        self.Bphase_j = Bphase_j
        self.amp_function_list = amp_function_list
        self.phase_function_list = phase_function_list
        # TODO: Check for correct input.

    def evaluate(self, params):
        """Evaluate the reduced order model in dimensionless units.
        
        Parameters
        ----------
        params : array of floats
            The physical parameters of the ROM.
            
        Returns
        -------
        waveform : TimeDomainWaveform
            The ROM waveform.
        """
        return reduced_order_model_time_domain_waveform(params, self.Bamp_j, self.Bphase_j, 
                                                        self.amp_function_list, self.phase_function_list)
    
    def evaluate_physical_units(self, phys_params):
        """Evaluate the waveform with physics time and strain units, including 
        inclination, f_low, etc.
        """
        # TODO: make this function
        pass


def load_reduced_order_model_time_domain_waveform(Bamp_filename, Bphase_filename, memb_size,
                                                  ampcoeff_filename, phasecoeff_filename):
    """
    """
    # Extract lists of Bamp and Bphase interpolating functions(time)
    Bamp = tdwave.load_hdf5_time_domain_waveform_set(Bamp_filename, memb_size=memb_size)
    Bphase = tdwave.load_hdf5_time_domain_waveform_set(Bphase_filename, memb_size=memb_size)
    Bamp_list = [Bamp.get_waveform_data(i) for i in range(Bamp.get_len())]
    Bphase_list = [Bphase.get_waveform_data(i) for i in range(Bphase.get_len())]
    Bamp.close()
    Bphase.close()
    
    # Extract lists of coefficients for the interpolating functions(params)
    amp_coeff_list, params_min, params_max = cheb.load_chebyshev_coefficients_list(ampcoeff_filename)
    phase_coeff_list, params_min, params_max = cheb.load_chebyshev_coefficients_list(phasecoeff_filename)
    
    # Generate the amplitude and phase functions(params)
    amp_function_list = [cheb.chebyshev_interpolation3d_generator(amp_coeff_list[i], params_min, params_max) 
                         for i in range(len(amp_coeff_list))]
    phase_function_list = [cheb.chebyshev_interpolation3d_generator(phase_coeff_list[i], params_min, params_max) 
                           for i in range(len(phase_coeff_list))]
    
    return ReducedOrderModelTimeDomainWaveform(Bamp_list, Bphase_list, amp_function_list, phase_function_list)





