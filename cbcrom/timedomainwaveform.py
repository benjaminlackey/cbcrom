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
TimeDomainWaveform and TimeDomainWaveformSet classes, and associated functions.
"""

import numpy as np
import scipy.interpolate
import scipy.integrate

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.pylab import subplots_adjust

import copy
import h5py

# TimeSeries
import pycbc.types

import empiricalinterpolation as eim

################################################################################
#                 TimeDomainWaveform class and utilities                       #
################################################################################

class TimeDomainWaveform:
    """Methods for efficiently storing and resampling a complex, time-domain waveform.
    
    Attributes
    ----------
    time : numpy array
    amp : numpy array
    phase : numpy array
    """
    
    def __init__(self, time, amp, phase):
        """Initialize time, amp, phase.
        
        Parameters
        ----------
        time : numpy array
        amp : numpy array
        phase : numpy array
        """
        # Check that time, amp, phase have same length.
        if len(amp)!=len(time) or len(phase)!=len(time):
            raise Exception, 'time, amp, phase must have the same length.'
        # Check that time is monotonically increasing.
        if any([time[i]>=time[i+1] for i in range(len(time)-1)]):
            raise Exception, 'time must be monotonically increasing with no duplicates.'
                    
        self.time = copy.copy(time)
        self.amp = copy.copy(amp)
        self.phase = copy.copy(phase)
    
    def copy(self):
        """Make a copy of the waveform, 
        so that when you mess with the waveform you don't overwrite the original.
        """
        return TimeDomainWaveform(self.time, self.amp, self.phase)

    def get_complex(self):
        """Get the waveform in complex form amp*exp(i*phase).
        
        Returns
        -------
        h_complex : numpy array
        """
        return self.amp*np.exp(1.0j*self.phase)
        
    def time_shift(self, add_time):
        """Add add_t to the time.
        """
        self.time += add_time
            
    def phase_shift(self, add_phase=None, remove_start_phase=False):
        """Add add_phase to the phase,
        or zero the phase at the start.
        """
        # Add the phase add_phase
        if add_phase is not None:
            self.phase += add_phase
        # Shift the phase to be 0.0 at the first data point
        if remove_start_phase:
            self.phase += -self.phase[0]
            
    def add_point(self, t, order=2):
        """Calculate the point (t, amp(t), phase(t)) with interpolation,
        and add it to the data arrays.
        """
        # Check that t_start and t_end are in the range of self.time
        if t<self.time[0] or t>self.time[-1]:
            raise Exception, 'Time t must be in the range of self.time' 
            
        # Interpolate amp(time) and phase(time)
        ampoft = scipy.interpolate.UnivariateSpline(self.time, self.amp, k=order, s=0)
        phaseoft = scipy.interpolate.UnivariateSpline(self.time, self.phase, k=order, s=0)
        a = ampoft(t)
        p = phaseoft(t)
        
        # Insert the new point just before i_insert
        i_insert = next(i for i in range(len(self.time)) if t<self.time[i])
        self.time = np.insert(self.time, i_insert, t)
        self.amp = np.insert(self.amp, i_insert, a)
        self.phase = np.insert(self.phase, i_insert, p)
    
    def remove_decreasing_phase(self):
        """Remove data from the end of the waveform if the phase starts to decrease.
        """
        if any([self.phase[i]>=self.phase[i+1] for i in range(len(self.phase)-1)]):
            i_end_mono = next(i for i in range(len(self.phase)-1) if self.phase[i]>=self.phase[i+1])
            self.time = self.time[:i_end_mono+1]
            self.amp = self.amp[:i_end_mono+1]
            self.phase = self.phase[:i_end_mono+1]
            t_end = self.time[i_end_mono]
    
    def resample_at_times(self, time_new, order=2):
        """Resample the waveform at specific times in the array time_new.
        """
        # Check that t_start and t_end are in the range of self.time
        if time_new[0]<self.time[0] or time_new[-1]>self.time[-1]:
            raise Exception, 'The array time_new must be in the range of self.time.'

        # Interpolate amp(time) and phase(time)
        ampoft = scipy.interpolate.UnivariateSpline(self.time, self.amp, k=order, s=0)
        phaseoft = scipy.interpolate.UnivariateSpline(self.time, self.phase, k=order, s=0)
        self.time = time_new
        self.amp = ampoft(time_new)
        self.phase = phaseoft(time_new)

    def resample_uniform_in_time(self, t_start=None, t_end=None, delta_t=1.0, order=2):
        """Resample the waveform to be uniform in time.
        """
        # Check that t_start and t_end are in the range of self.time
        if t_start<self.time[0] or t_end>self.time[-1]:
            raise Exception, 'Start and end times must be in the range of self.time.'
        if t_start is None: t_start = self.time[0]
        if t_end is None: t_end = self.time[-1]
        
        time_new = np.arange(t_start, t_end, delta_t)
        self.resample_at_times(time_new, order=order)
        
    def resample_uniform_in_phase(self, t_start=None, t_end=None, samples_per_cycle=2.0, order=2):
        """Resample the waveform to be uniform in phase.
        The samples_per_cycle is approximate, 
        so that the first and last cycles are EXACTLY at t_start and t_end.
        """
        # Remove points at the end where the phase is decreasing.
        self.remove_decreasing_phase()
        
        # Check that t_start and t_end are in the range of self.time
        if t_start<self.time[0] or t_end>self.time[-1]:
            raise Exception, 'Start and end times must be in the range of self.time.'
        if t_start is None: t_start = self.time[0]
        if t_end is None: t_end = self.time[-1]
        
        # Interpolate time(phase)
        tofphase = scipy.interpolate.UnivariateSpline(self.phase, self.time, k=order, s=0)
        ampoft = scipy.interpolate.UnivariateSpline(self.time, self.amp, k=order, s=0)
        phaseoft = scipy.interpolate.UnivariateSpline(self.time, self.phase, k=order, s=0)

        # Uniform phase samples
        phi_start = phaseoft(t_start)
        phi_end = phaseoft(t_end)
        npoints = int(np.ceil((phi_end - phi_start)*samples_per_cycle/(2.0*np.pi))) + 1
        self.phase = np.linspace(phi_start, phi_end, npoints)
        
        # time, amp samples
        self.time = tofphase(self.phase)
        self.time[0] = t_start # Make sure this is exactly as requested (not just approximate)
        self.time[-1] = t_end # Make sure this is exactly as requested (not just approximate)
        self.amp = ampoft(self.time)

    # !!!!!!! Deprecated: !!!!!!!
    def resample(self, t_start=None, t_end=None, delta_t=None, samples_per_cycle=None, time_new=None, order=2):
        """Resample the waveform with various methods.
        """
        if t_start is None: t_start = self.time[0]
        if t_end is None: t_end = self.time[-1]
        
        # Check that t_start and t_end are in the range of self.time
        if t_start<self.time[0] or t_end>self.time[-1]:
            raise Exception, 't_start and t_end must be in the range of self.time' 
        
        # Interpolate amp(time) and phase(time)
        ampoft = scipy.interpolate.UnivariateSpline(self.time, self.amp, k=order, s=0)
        phaseoft = scipy.interpolate.UnivariateSpline(self.time, self.phase, k=order, s=0)
        
        # Uniform in time sampling (won't necessarily include t_end)
        if delta_t is not None:
            self.time = np.arange(t_start, t_end, delta_t)
            self.amp = ampoft(self.time)
            self.phase = phaseoft(self.time)
            
        # Uniform in phase sampling (guaranteed to include t_end)
        if samples_per_cycle is not None:
            if any([self.phase[i]>=self.phase[i+1] for i in range(len(self.phase)-1)]):
                #print 'Warning: phase is not monotonically increasing with no duplicates.\n'+\
                #    'Truncating points after phase first stops being monotonically increasing.'
                i_end_mono = next(i for i in range(len(self.phase)-1) if self.phase[i]>=self.phase[i+1])
                self.time = self.time[:i_end_mono+1]
                self.amp = self.amp[:i_end_mono+1]
                self.phase = self.phase[:i_end_mono+1]
                t_end = self.time[i_end_mono]     
            tofphase = scipy.interpolate.UnivariateSpline(self.phase, self.time, k=order, s=0)
            # Uniform phase samples
            phi_start = phaseoft(t_start)
            phi_end = phaseoft(t_end)
            npoints = int(np.ceil((phi_end - phi_start)*samples_per_cycle/(2.0*np.pi))) + 1
            self.phase = np.linspace(phi_start, phi_end, npoints)
            # time, amp samples
            self.time = tofphase(self.phase)
            self.time[0] = t_start # Make sure this is exactly as requested (not just approximate)
            self.time[-1] = t_end # Make sure this is exactly as requested (not just approximate)
            self.amp = ampoft(self.time)
        
        # sample the waveform at the specific times time
        if time_new is not None:
            # Check that t_start and t_end are in the range of self.time
            if time_new[0]<self.time[0] or time_new[-1]>self.time[-1]:
                raise Exception, 'time must be in the range of self.time'
            self.time = time_new
            self.amp = ampoft(self.time)
            self.phase = phaseoft(self.time)
        
    def pycbc_time_series(self, delta_t, order=2):
        """Generate pycbc TimeSeries.
        """
        # Make a copy of the waveform so you don't resample the waveform itself
        h = self.copy()
        h.resample(delta_t=delta_t, order=order)
        hcomp = h.get_complex()
        return pycbc.types.TimeSeries(hcomp, epoch=h.time[0], delta_t=delta_t)


def complex_to_time_domain_waveform(time, hcomplex):
    """Convert complex numpy array to TimeDomainWaveform.
    
    Parameters
    ----------
    time : numpy array
    hcomplex : complex numpy array
    
    Returns
    -------
    h : TimeDomainWaveform
    """
    hamp = np.abs(hcomplex)
    hphase = np.angle(hcomplex)
    hphase = np.unwrap(hphase)
    return TimeDomainWaveform(time, hamp, hphase)


def resample_waveforms_to_match_start_end_times(h1, h2, samples_per_cycle=100.0):
    """Resample h1 and h2 so they have the same time samples.
    This function assumes that t=0 corresponds to the maximum amplitude.
    
    Parameters
    ----------
    h1 : TimeDomainWaveform
    h2 : TimeDomainWaveform
    samples_per_cycle : float, 100
    
    Returns
    -------
    h1_new : TimeDomainWaveform
    h2_new : TimeDomainWaveform
    """
    # Copy the waveforms so you don't overwrite them.
    h1_new = h1.copy()
    h2_new = h2.copy()
    
    # Get new start and end times that are common to both waveforms
    t_start = max(h1_new.time[0], h2_new.time[0])
    t_end = min(h1_new.time[-1], h2_new.time[-1])
    
    # Sample evenly in phase
    h1_new.resample(t_start=t_start, t_end=t_end, samples_per_cycle=samples_per_cycle)
    h2_new.resample(time_new=h1_new.time)
    
    # Shift phase to be 0 at beginning for both
    h1_new.phase_shift(remove_start_phase=True)
    h2_new.phase_shift(remove_start_phase=True)
    
    return h1_new, h2_new


################################################################################
#               Plotting and comparing TimeDomainWaveforms                     #
################################################################################

def plot_time_domain_waveform(fig, waveform, imag=False, mag=False, 
                              xlim=None, xlabel=r'$tc^3/GM$',
                              ylabel_pol=r'$h_+ + i h_\times$', 
                              ylabel_amp=r'$A$', ylabel_phase=r'$\Phi$', 
                              pol_legend=True, wave_legend=False):
    """Plot the amplitude, phase, and polarizations of a waveform.
    """
    # Polarization plot
    axes = fig.add_subplot(311)
    t = waveform.time
    hcomp = waveform.get_complex()
    label = r'$h_+$' if pol_legend else ''
    line_list = axes.plot(t, hcomp.real, ls='-', label=label)
    color = line_list[0].get_color()
    if imag:
        label = r'$h_\times$' if pol_legend else ''
        axes.plot(t, hcomp.imag, ls='--', c=color, label=label)
    if mag:
        label = r'$|h_+ + ih_\times|$' if pol_legend else ''
        axes.plot(waveform.time, waveform.amp, ls=':', c=color, label=label)
    
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_ylabel(ylabel_pol, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    axes.legend(fontsize=14, loc='best', ncol=3)    
    
    # Amplitude plot
    axes = fig.add_subplot(312)
    axes.plot(waveform.time, waveform.amp, c=color)
    
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_ylabel(ylabel_amp, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    
    # Phase plot
    axes = fig.add_subplot(313)
    label = wave_legend if wave_legend is not False else ''
    axes.plot(waveform.time, waveform.phase, c=color, label=label)

    if xlim is not None: axes.set_xlim(xlim)
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel_phase, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.legend(fontsize=14, loc='best', ncol=2)
    
    subplots_adjust(hspace=0.07)
    
    
def plot_time_domain_waveform_list(fig, waveforms, imag=False, mag=False, 
                                   xlim=None, xlabel=r'$tc^3/GM$',
                                   ylabel_pol=r'$h_+ + i h_\times$', 
                                   ylabel_amp=r'$A$', ylabel_phase=r'$\Phi$', 
                                   pol_legend=True, wave_legend=False):
    """Plot the amplitude, phase, and polarizations for a list of waveforms.
    
    Parameters
    ----------
    waveforms : TimeDomainWaveform or a list of them
    """
    if type(waveforms) is not list: waveforms = [waveforms]
    Nwave = len(waveforms)

    for i in range(Nwave):
        pol_legend = True if i==0 else False
        wl = str(i) if wave_legend is False else wave_legend
        
        plot_time_domain_waveform(fig, waveforms[i], imag=imag, mag=mag,
                                  xlim=xlim, xlabel=xlabel,
                                  ylabel_pol=ylabel_pol, 
                                  ylabel_amp=ylabel_amp, ylabel_phase=ylabel_phase, 
                                  pol_legend=pol_legend, wave_legend=wl)


def plot_time_domain_waveform_difference(fig, h1, h2, imag=False, mag=False, 
                                         xlim=None, xlabel=r'$tc^3/GM$',
                                         h1_legend=r'$h_1$', h2_legend=r'$h_2$', 
                                         ylabel_pol=r'$h = h_+ + i h_\times$', 
                                         ylabel_amp=r'$A_1/A_2 - 1$', 
                                         ylabel_phase=r'$\Phi_1 - \Phi_2$'):
    """Plot the amplitude, phase, and polarizations of a waveform.
    """
    # Plot the two waveforms
    axes = fig.add_subplot(311)
    t = h1.time
    h1comp = h1.get_complex()
    axes.plot(t, h1comp.real, color='b', ls='-', lw=1, label=h1_legend)
    axes.plot(t, h1comp.imag, color='b', ls='-', lw=1)
    axes.plot(t, np.abs(h1comp), color='b', ls='-', lw=1)
    
    t = h2.time
    h2comp = h2.get_complex()
    axes.plot(t, h2comp.real, color='r', ls='--', lw=1, label=h2_legend)
    axes.plot(t, h2comp.imag, color='r', ls='--', lw=1)
    axes.plot(t, np.abs(h2comp), color='r', ls='--', lw=1)
    
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_ylabel(ylabel_pol, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    axes.legend(fontsize=14, loc='best', ncol=3)    
    
    # Amplitude plot
    axes = fig.add_subplot(312)
    error = h1.amp/h2.amp-1.0
    axes.plot(h1.time, error, color='b', ls='-', lw=1)
    axes.plot([h1.time[0], h1.time[-1]], [0.0, 0.0], color='k', ls=':', lw=1)
    
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_ylabel(ylabel_amp, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    
    # Phase plot
    axes = fig.add_subplot(313)
    error = h1.phase - h2.phase
    axes.plot(h1.time, error, color='b', ls='-', lw=1)
    axes.plot([h1.time[0], h1.time[-1]], [0.0, 0.0], color='k', ls=':', lw=1)
    
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel_phase, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.legend(fontsize=14, loc='best', ncol=2)
    
    subplots_adjust(hspace=0.07)


################################################################################
#                    HDF5TimeDomainWaveformSet class                           #
################################################################################

class HDF5TimeDomainWaveformSet:
    """Methods for reading and writing a set of TimeDomainWaveform objects 
    from an hdf5 file
    that contains a list of TimeSeries data.
    
    Attributes
    ----------
    parameter_names : List of strings
        The names of the waveform parameters.
    grid_shape : List of ints
        Shape of the grid.
    ws_file : h5py.File object
        Pointer to the hdf5 file.
    """
    
    def __init__(self, filename, mode='r', memb_size=2**31-1):
        """Get a list of waveforms and data associated with those waveforms.
        
        Parameters
        ----------
        filename : string
            Name of the hdf5 file to store the TimeDomainWaveformSet
        """
        self.parameter_names = None
        self.grid_shape = None
        # Read/write if file exists, create otherwise
        #self.ws_file = h5py.File(filename, 'a', libver='latest')
        #self.ws_file = h5py.File(filename, mode, driver='family', memb_size=memb_size, libver='latest')
        self.ws_file = h5py.File(filename, mode, driver='family', memb_size=memb_size)

    def set_parameter_names(self, names):
        """
        names: List of strings
            The names of the waveform parameters.
        """
        self.parameter_names = names
        self.ws_file['parameter_names'] = names
        
    def set_grid_shape(self, shape=None):
        """
        shape : List like
            Shape of the grid if the waveforms are constructed from a regular grid of parameters.
            Ex. For a list of 125 waveforms constructed by varying 3 parameters over 5 values each,
            the shape would be [5, 5, 5].
        """
        if shape is not None:
            self.grid_shape = list(shape)
            self.ws_file['shape'] = shape
        else:
            self.grid_shape = None
            self.ws_file['shape'] = [0]

    def close(self):
        """Close the hdf5 file.
        """
        self.ws_file.close()
    
    def add_waveform(self, waveform, parameters, index):
        """Add a waveform to the hdf5 file.
        
        Parameters
        ----------
        waveform : TimeDomainWaveform
        parameters : List
            Parameters of the waveform
        index : int
            For index i, the waveform will be stored under the group 'wave_i'.
        """
        groupname = 'wave_'+str(index)
        wave = self.ws_file.create_group(groupname)
        wave['parameters'] = parameters
        wave['time'] = waveform.time
        wave['amplitude'] = waveform.amp
        wave['phase'] = waveform.phase
    
    def overwrite_waveform(self, waveform, parameters, index):
        """Delete a waveform from the hdf5 file, and write over it with a new waveform.
        The waveform can have a different length.
        
        Parameters
        ----------
        waveform : TimeDomainWaveform
        parameters : List
            Parameters of the waveform
        index : int
            For index i, the waveform will be stored under the group 'wave_i'.
        """
        # Delete waveform
        groupname = 'wave_'+str(index)
        del self.ws_file[groupname]
        # Add new waveform with same group name
        self.add_waveform(waveform, parameters, index)
    
    def add_waveform_list(self, waveform_list, parameters_list):
        """Add a list of N waveforms to the hdf5 file.
        Waveforms will be stored under the groups 'wave_0'--'wave_N.
        
        Parameters
        ----------
        waveform_list : List of TimeDomainWaveform
        parameters_list : List of parameters of the waveform
        """
        Nwaves = len(waveform_list)
        for i in range(Nwaves):
            self.add_waveform(waveform_list[i], parameters_list[i], i)

    def get_waveform_data(self, index, data='waveform'):
        """Load a single complex TimeSeries waveform from the HDF5 file.
            
        Parameters
        ----------
        index : int
            Index of the waveform you want.
        data : str, {'waveform', parameters}
            The data to extract for the waveform.
    
        Returns
        -------
        TimeDomainWaveform for 'waveform'
        array of floats for 'parameters'
        """
        # Get the waveform group
        groupname = 'wave_'+str(index)
        wave = self.ws_file[groupname]
        if data == 'waveform':
            return TimeDomainWaveform(wave['time'][:], wave['amplitude'][:], wave['phase'][:])
        elif data == 'parameters':
            return wave['parameters'][:]
        else:
            raise Exception, 'Valid data options are {waveform, parameters}.'
    
    def get_len(self):
        """Get the number of waveforms in the waveform set.
        """
        names = self.ws_file.keys()
        wavegroups = [names[i] for i in range(len(names)) if 'wave_' in names[i]]
        return len(wavegroups)
        
    def get_parameters(self):
        """Get a list of the waveform parameters.
        
        Returns
        -------
        parameters : 2d list
            List of waveform parameters.
        """
        Nwaves = self.get_len()
        return [list(self.get_waveform_data(i, data='parameters')) for i in range(Nwaves)]

    def get_parameter_grid(self):
        """
        Get the parameters in a grid of dimension Nparams.
        
        Returns
        -------
        parameter_grid : (N+1)d array
            Grid of parameters for N dimensional parameter space.
        """
        if self.grid_shape is not None:
            Nparams = len(self.get_waveform_data(0, data='parameters'))
            parameters = self.get_parameters()
            return np.array(parameters).reshape(tuple(self.grid_shape+[Nparams]))
        else:
            raise Exception, 'grid_shape not provided.'


def load_hdf5_time_domain_waveform_set(filename, memb_size=2**31-1):
    """Create an HDF5TimeSeriesSet object from a file.
    """
    # Create object and open file
    ws = HDF5TimeDomainWaveformSet(filename, mode='r+', memb_size=memb_size)
    # Set parameter_names form file data
    ws.parameter_names = list(ws.ws_file['parameter_names'][:])
    # Set grid_shape from file data
    ws.grid_shape = list(ws.ws_file['shape'][:])
    if ws.grid_shape[0] == 0:
        ws.grid_shape = None
    
    return ws


################################################################################
#                  Arithmetic for TimeDomainWaveform objects                   #
#                        (h1+h2, h1-h2, alpha*h, h1.h2)                        #
################################################################################


# def check_waveform_consistency(h1, h2):
#     """Check that h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
#     """
#     if h1.delta_t != h2.delta_t:
#         raise Exception, 'h1.delta_t='+str(h1.delta_t)+' and h2.delta_t='+str(h1.delta_t)+' are not the same.'
#     if h1.start_time != h2.start_time:
#         raise Exception, 'h1.start_time='+str(h1.delta_t)+' and h2.start_time='+str(h1.delta_t)+' are not the same.'
#     if len(h1) != len(h2):
#         raise Exception, 'len(h1)='+str(len(h1))+' and len(h1)='+str(len(h1))+' are not the same.'


def add(h1, h2):
    """Evaluate h1+h2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    hcomp = h1.get_complex() + h2.get_complex()
    hamp = np.abs(hcomp)
    hphase = np.angle(hcomp)
    hphase = np.unwrap(hphase)
    return TimeDomainWaveform(h1.time, hamp, hphase)


def subtract(h1, h2):
    """Evaluate h1-h2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    hcomp = h1.get_complex() - h2.get_complex()
    hamp = np.abs(hcomp)
    hphase = np.angle(hcomp)
    hphase = np.unwrap(hphase)
    return TimeDomainWaveform(h1.time, hamp, hphase)


def scalar_multiply(alpha, h):
    """Multiply a waveform h by a float alpha
    """
    alpha_amp = np.abs(alpha)
    alpha_phase = np.angle(alpha)
    return TimeDomainWaveform(h.time, alpha_amp*h.amp, alpha_phase+h.phase)


def inner_product(h1, h2):
    """Evaluate the inner product < h1, h2 > = int_tL^tH dt h1*(t) h2(t).
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.get_complex().conj()*h2.get_complex()
    diff = np.array([h1.time[i+1]-h1.time[i] for i in range(len(h1.time)-1)])
    sum_neighbors = np.array([integrand[i]+integrand[i+1] for i in range(len(integrand)-1)])
    #return 0.5*np.dot(diff, sum_neighbors)
    return 0.5*np.sum( diff * sum_neighbors )


################################################################################
#          Arithmetic for amplitude of TimeDomainWaveform objects              #
#                        (h1+h2, h1-h2, alpha*h, h1.h2)                        #
################################################################################


def add_amp(h1, h2):
    """Evaluate |h1|+|h2|.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    return TimeDomainWaveform(h1.time, h1.amp+h2.amp, np.zeros(len(h1.time)))


def subtract_amp(h1, h2):
    """Evaluate |h1|-|h2|.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    return TimeDomainWaveform(h1.time, h1.amp-h2.amp, np.zeros(len(h1.time)))


def scalar_multiply_amp(alpha, h):
    """Multiply the |h| by a float alpha
    """
    return TimeDomainWaveform(h.time, alpha*h.amp, np.zeros(len(h.time)))


def inner_product_amp(h1, h2):
    """Evaluate the inner product < |h1|, |h2| > = int_tL^tH dt |h1(t)| |h2(t)|.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.amp*h2.amp
    diff = np.array([h1.time[i+1]-h1.time[i] for i in range(len(h1.time)-1)])
    sum_neighbors = np.array([integrand[i]+integrand[i+1] for i in range(len(integrand)-1)])
    return 0.5*np.sum( diff * sum_neighbors )


def inner_product_amp_simps(h1, h2):
    """Use Simpson's rule to
    evaluate the inner product < |h1|, |h2| > = int_tL^tH dt |h1(t)| |h2(t)|.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.amp*h2.amp
    return scipy.integrate.simps(integrand, x=h1.time)


def inner_product_amp_samples(h1, h2):
    """Sum the product of h1 and h2 at each sample.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.amp*h2.amp
    return np.sum( integrand )


################################################################################
#          Arithmetic for phase of TimeDomainWaveform objects                  #
#                        (h1+h2, h1-h2, alpha*h, h1.h2)                        #
################################################################################


def add_phase(h1, h2):
    """Evaluate phi1+phi2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    return TimeDomainWaveform(h1.time, np.zeros(len(h1.time)), h1.phase+h2.phase)


def subtract_phase(h1, h2):
    """Evaluate phi1-phi2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    return TimeDomainWaveform(h1.time, np.zeros(len(h1.time)), h1.phase-h2.phase)


def scalar_multiply_phase(alpha, h):
    """Multiply the phase of h by a float alpha
    """
    return TimeDomainWaveform(h.time, np.zeros(len(h.time)), alpha*h.phase)


def inner_product_phase(h1, h2):
    """Evaluate the inner product < phi1, phi2 > = int_tL^tH dt phi1(t) phi2(t).
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.phase*h2.phase
    diff = np.array([h1.time[i+1]-h1.time[i] for i in range(len(h1.time)-1)])
    sum_neighbors = np.array([integrand[i]+integrand[i+1] for i in range(len(integrand)-1)])
    return 0.5*np.sum( diff * sum_neighbors )


def inner_product_phase_simps(h1, h2):
    """Use Simpson's rule to
    evaluate the inner product < phi1, phi2 > = int_tL^tH dt phi1(t) phi2(t).
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.phase*h2.phase
    return scipy.integrate.simps(integrand, x=h1.time)


def inner_product_phase_samples(h1, h2):
    """Sum the product of h1 and h2 at each sample.
    """
    #check_waveform_consistency(h1, h2)
    integrand = h1.phase*h2.phase
    return np.sum( integrand )




