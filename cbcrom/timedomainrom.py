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

# Use the exact same units as LAL
G_SI = 6.67384e-11
C_SI = 299792458.0
MPC_SI = 3.085677581491367e+22
MSUN_SI = 1.9885469549614615e+30

################################################################################
#              Empirical interpolation for TimeDomainWaveforms                 #
################################################################################

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


def reduced_order_model_physical_units(Bamp_j, Bphase_j, amp_function_list, phase_function_list,
                                       mass1=None, mass2=None, lambda1=None, lambda2=None,
                                       delta_t=None, f_lower=None, f_ref=None,
                                       distance=None, inclination=None, coa_phase=None):
    """Construct a waveform with pycbc units from a ROM with dimensionless units.
    
    Parameters
    ----------
    mass1 : float
        The mass of the first component object in the binary in solar masses.
    mass2 :
        The mass of the second component object in the binary in solar masses.
    delta_t :
        The time step used to generate the waveform.
    f_lower :
        The starting frequency of the waveform.
    f_ref : {float}, optional
        The reference frequency
    distance : {1, float}, optional
        The distance from the observer to the source in megaparsecs.
    inclination : {0, float}, optional
        The inclination angle of the source.
    coa_phase : {0, float}, optional
        The final phase or phase at the peak of the wavform. See documentation
        on specific approximants for exact usage.
    lambda1: {0, float}, optional
        The tidal deformability parameter of object 1.
    lambda2: {0, float}, optional
        The tidal deformability parameter of object 2.
    
    Returns
    -------
    tstart : float
    delta_t : float
    hplus : array
    hcross : array
    """
    ################ Checking for correct input ################
    if f_lower < 10.0 or f_lower>700.0: raise ValueError('f_lower must be in range [11, 700]Hz')
    
    if mass1 < 1.0 or mass1 > 2.0 or mass2 < 1.0 or mass2 > 2.0:
        raise ValueError('Valid mass range: mass1 in [1, 2], mass2 in [1, 2]')
    if lambda1 < 50.0 or lambda1 > 5000.0 or lambda2 < 50.0 or lambda2 > 5000.0:
        raise ValueError('Valid tidal parameter range: lambda1 in [50, 5000], lambda2 in [50, 5000]')
    
    # Swap (mass1, mass2) and (lambda1, lambda2) if mass1 is not the larger mass
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        lambda1, lambda2 = lambda2, lambda1
    
    q = mass2/mass1
    if q > 1.0: raise Exception, 'q must be <= 1.'
    
    ########## Constructing resampled waveform with physical units #########
    params = [q, lambda1, lambda2]
    h = reduced_order_model_time_domain_waveform(params, Bamp_j, Bphase_j,
                                                 amp_function_list, phase_function_list)
       
    # Get times in seconds
    mtot = (mass1+mass2)*MSUN_SI
    time_phys = h.time*G_SI*mtot/C_SI**3
    eta = mass1*mass2/(mass1+mass2)**2

    order = 3
    ampoft = scipy.interpolate.UnivariateSpline(time_phys, h.amp, k=order, s=0)
    phaseoft = scipy.interpolate.UnivariateSpline(time_phys, h.phase, k=order, s=0)
    omegaoft = phaseoft.derivative(n=1)
    freq = omegaoft(time_phys)/(2*np.pi)
    #print freq[0], freq[-1]
    
    # Find region where frequency is monotonically increasing, then construct t(f)
    i_end_mono = next( (i for i in range(len(freq)-1) if freq[i]>=freq[i+1]), (len(freq)-1) )
    toffreq = scipy.interpolate.UnivariateSpline(freq[:i_end_mono], time_phys[:i_end_mono], k=order, s=0)
    
    # Resample with even spacing
    tstart = toffreq([f_lower])[0]
    time_phys_res = np.arange(tstart, time_phys[-1], delta_t)
    amp_res = ampoft(time_phys_res)
    phase_res = phaseoft(time_phys_res)
    # Zero the phase at the beginning
    phase_res -= phase_res[0]
    
    # Rescale amplitude
    #h22_to_h = np.sqrt(5.0/np.pi)/8.0
    #amp_units = G_SI*mtot/(C_SI**2*distance*MPC_SI)
    #amp_rescale = amp_units*h22_to_h*amp_res
    h22_to_h = 4.0*eta*np.sqrt(5.0/np.pi)/8.0
    amp_units = G_SI*mtot/(C_SI**2*distance*MPC_SI)
    amp_rescale = amp_units*h22_to_h*amp_res

    # Adjust for inclination angle [0, pi]
    inc_plus = (1.0+np.cos(inclination)**2)/2.0
    inc_cross = np.cos(inclination)
    
    hplus = inc_plus*amp_rescale*np.cos(phase_res)
    hcross = inc_cross*amp_rescale*np.sin(phase_res)
    return tstart, delta_t, hplus, hcross


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
    
    def evaluate_physical_units(self, mass1=None, mass2=None, lambda1=None, lambda2=None,
                                delta_t=None, f_lower=None, f_ref=None,
                                distance=None, inclination=None, coa_phase=None):
        """Evaluate the waveform with physical units of time and strain.
        """
        return reduced_order_model_physical_units(self.Bamp_j, self.Bphase_j,
                                                  self.amp_function_list, self.phase_function_list,
                                                  mass1=mass1, mass2=mass2, lambda1=lambda1, lambda2=lambda2,
                                                  delta_t=delta_t, f_lower=f_lower, f_ref=f_ref,
                                                  distance=distance, inclination=inclination, coa_phase=coa_phase)


def load_reduced_order_model_time_domain_waveform(Bamp_filename, Bphase_filename, memb_size,
                                                  ampcoeff_filename, phasecoeff_filename, 
                                                  logamp=False, logphase=False):
    """Load a ROM from hdf5 files.
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
    amp_function_list = [cheb.chebyshev_interpolation3d_generator(amp_coeff_list[i], params_min, params_max, log=logamp) 
                         for i in range(len(amp_coeff_list))]
    phase_function_list = [cheb.chebyshev_interpolation3d_generator(phase_coeff_list[i], params_min, params_max, log=logphase) 
                           for i in range(len(phase_coeff_list))]
    
    return ReducedOrderModelTimeDomainWaveform(Bamp_list, Bphase_list, amp_function_list, phase_function_list)





