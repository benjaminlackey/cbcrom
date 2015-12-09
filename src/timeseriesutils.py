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
Provides utilities for pycbc TimeSeries waveforms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.pylab import subplots_adjust
import pycbc.types # TimeSeries
import pycbc.waveform # amplitude and phase functions


################################################################################
# Functions for generic list of TimeSeries
################################################################################

def evaluate_time_slice_over_grid(time_series_list, shape, time_index):
    """Take a list of TimeSeries that are constructed from a regular grid.
    Evaluate the TimeSeries at the index time_index.
    Then construct a grid from that quantity.
    
    Parameters
    ----------
    time_series_list : List of TimeSeries
    shape : tuple like list of length n
        The number of points in each of the n dimensions.
    time_index : int
        Index for the sample of each TimeSeries.
    
    Returns
    -------
    grid : nd-array
    """
    return np.array([time_series_list[j][time_index] for j in range(len(time_series_list))]).reshape(tuple(shape))


################################################################################
#         Functions to convert between (complex h), (amp, phase).              #
################################################################################


def amplitude_from_complex_time_series(h):
    """Return the amplitude of the complex TimeSeries h.
    
    Parameters
    ----------
    h : TimeSeries
        A complex PyCBC TimeSeries.
    
    Returns
    -------
    phase : TimeSeries
        A real PyCBC TimeSeries containing the amplitude.
    """
    # Calculate the amplitude for each point.
    amp = np.abs(h.numpy())
    
    # Return the amplitude as a real TimeSeries
    return pycbc.types.TimeSeries(amp, delta_t=h.delta_t, epoch=h.start_time)


def phase_from_complex_time_series(h, remove_start_phase=True):
    """Return the unwrapped phase.
    
    Parameters
    ----------
    h : TimeSeries
        A complex PyCBC TimeSeries.
    
    Returns
    -------
    phase : TimeSeries
        A real PyCBC TimeSeries containing the unwrapped phase.
    """
    # Calculate the angle (between -pi and pi) for each point.
    phase_wrapped = np.angle(h.numpy())
    
    # Unwrap the phase
    phase = np.unwrap(phase_wrapped)
    
    # Shift the phase to be 0.0 at the first data point
    if remove_start_phase:
        phase += -phase[0]
    
    # Return the phase as a real TimeSeries
    return pycbc.types.TimeSeries(phase, delta_t=h.delta_t, epoch=h.start_time)


################################################################################
#     Functions to convert between (complex h), (hp, hc), (amp, phase).        #
################################################################################


def polarizations_to_complex(hp, hc):
    #!!!! Check that hp, hc have same delta_t, length, epoch, etc.

    epoch = hp.start_time
    delta_t = hp.delta_t
    return pycbc.types.TimeSeries(hp+1.0j*hc, epoch=epoch, delta_t=delta_t)


def complex_to_polarizations(hcomplex):
    """Take a TimeSeries of h_+ + ih_x.
    Return the real (hp) and imaginary (hc) parts as TimeSeries.
    """
    epoch = hcomplex.start_time
    delta_t = hcomplex.delta_t
    
    hp = pycbc.types.TimeSeries(hcomplex.numpy().real, epoch=epoch, delta_t=delta_t)
    hc = pycbc.types.TimeSeries(hcomplex.numpy().imag, epoch=epoch, delta_t=delta_t)
    
    return hp, hc


def amp_phase_from_complex(hcomplex, remove_start_phase=True):
    """Take a complex TimeSeries of h_+ + ih_x.
    Return two real TimeSeries representing the amplitude and unwrapped phase.
    """
    amp = amplitude_from_complex_time_series(hcomplex)
    phase = phase_from_complex_time_series(hcomplex, remove_start_phase=remove_start_phase)
    
    return amp, phase


#def amp_phase_from_complex(hcomplex, remove_start_phase=True):
#    """Take a TimeSeries of h_+ + ih_x.
#    Return two TimeSeries representing the amplitude and unwrapped phase.
#    """
#    epoch = hcomplex.start_time
#    delta_t = hcomplex.delta_t
#    
#    hp = pycbc.types.TimeSeries(hcomplex.numpy().real, epoch=epoch, delta_t=delta_t)
#    hc = pycbc.types.TimeSeries(hcomplex.numpy().imag, epoch=epoch, delta_t=delta_t)
#    
#    amp = pycbc.waveform.amplitude_from_polarizations(hp, hc)
#    phase = pycbc.waveform.phase_from_polarizations(hp, hc, remove_start_phase=remove_start_phase)
#    
#    return amp, phase


def complex_from_amp_phase(amp, phase):
    """Take a TimeSeries of h_+ + ih_x.
    Return two TimeSeries representing the amplitude and unwrapped phase.
    """
    epoch = amp.start_time
    delta_t = amp.delta_t
    
    hplus = amp.numpy()*np.cos(phase.numpy())
    hcross = amp.numpy()*np.sin(phase.numpy())
    
    return pycbc.types.TimeSeries(hplus+1.0j*hcross, epoch=epoch, delta_t=delta_t)


def amp_phase_from_polarizations(hp, hc, remove_start_phase=True):
    """Return two TimeSeries representing the amplitude and unwrapped phase.
    """
    
    amp = pycbc.waveform.amplitude_from_polarizations(hp, hc)
    phase = pycbc.waveform.phase_from_polarizations(hp, hc, remove_start_phase=remove_start_phase)
    
    return amp, phase


################################################################################
#                   Normalization and arithmetic functions                     #
################################################################################


def check_waveform_consistency(h1, h2):
    """Check that h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    if h1.delta_t != h2.delta_t:
        raise Exception, 'h1.delta_t='+str(h1.delta_t)+' and h2.delta_t='+str(h1.delta_t)+' are not the same.'
    if h1.start_time != h2.start_time:
        raise Exception, 'h1.start_time='+str(h1.delta_t)+' and h2.start_time='+str(h1.delta_t)+' are not the same.'
    if len(h1) != len(h2):
        raise Exception, 'len(h1)='+str(len(h1))+' and len(h1)='+str(len(h1))+' are not the same.'

def add_time_series(h1, h2):
    """Evaluate h1+h2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    check_waveform_consistency(h1, h2)
    return pycbc.types.TimeSeries(h1+h2, delta_t=h1.delta_t, epoch=h1.start_time)

def subtract_time_series(h1, h2):
    """Evaluate h1-h2.
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    check_waveform_consistency(h1, h2)
    return pycbc.types.TimeSeries(h1-h2, delta_t=h1.delta_t, epoch=h1.start_time)

def scalar_multiply_time_series(alpha, h):
    """Multiply a waveform h by a float alpha
    """
    return pycbc.types.TimeSeries(h*alpha, delta_t=h.delta_t, epoch=h.start_time)

def inner_product_time_series(h1, h2):
    """Evaluate the inner product < h1, h2 > = int_tL^tH dt h1*(t) h2(t).
    Assumes h1 and h2 have the same (1) length, (2) start_time, (3) delta_t.
    """
    check_waveform_consistency(h1, h2)
    integrand = np.sum( np.array(h1.numpy().tolist()).conj() * np.array(h2.numpy().tolist()) )
    inner = h1.delta_t * integrand
    return inner


################################################################################
#                       Functions for a list of waveforms                      #
################################################################################

def normalize_waveform_list(waveform_list):
    """Normalize a list of waveforms.
    
    Parameters
    ----------
    waveform_list : List of complex TimeSeries
        The waveforms of interest.
    
    Returns
    -------
    norm_list : List of floats
        List of the norms \sqrt{<h|h>}
    normalized_waveforms : List of complex TimeSeries
        Normalized waveforms.
    """
    norm_list = []
    normalized_waveforms = []
    for i in range(len(waveform_list)):
        norm, hnormalized = normalize_waveform(waveform_list[i])
        norm_list.append(norm)
        normalized_waveforms.append(hnormalized)
    
    return norm_list, normalized_waveforms


def align_waveform_set_at_max_amplitude(waveform_set):
    """Align waveforms at max amplitude and truncate the beginnings so they all start at the same time.
    Set phase at new beginning to be zero.
    Set length to power of 2, padding the end with zeros.
    
    Parameters
    ----------
    waveform_list : List of complex TimeSeries
        The waveforms of interest.
    
    Returns
    -------
    aligned_waveforms : List of complex TimeSeries
        Aligned waveforms.
    """
    # Get index of max amplitude for each waveform
    imax_list = []
    for i in range(len(waveform_set.waveforms)):
        ampmax, imax = waveform_set.waveforms[i].abs_max_loc()
        imax_list.append(imax)

    # Find the earliest time for the max amplitude (i.e. the minimum value of imaxlist)
    iearliest = min(imax_list)

    ###### Resize the waveforms so they are all the same power of 2 points ######

    # Find longest waveform
    Nmax = max([len(waveform_set.waveforms[i].numpy()) for i in range(len(waveform_set.waveforms))])
    
    # Find power of 2 just greater than Nmax
    pow2 = 0
    while 2**pow2<Nmax:
        pow2 +=1
    Nmax = 2**pow2
    
    ##### Shift each waveform so that the max of each waveform is the same as that of the iearliest waveform. #####
    # This will truncate the beginning of all waveforms except the iearliest
    adjusted_data_list = []
    for i in range(len(waveform_set.waveforms)):
        data = waveform_set.waveforms[i].numpy()
        
        # Allocate memory for resized waveform
        adjusted_data = np.zeros(Nmax, dtype=complex)
        # Shift waveform
        adjusted_data[:iearliest+1] = data[imax_list[i]-iearliest:]
        # Shift phase to be zero at start
        phase0 = np.angle(adjusted_data[0])
        adjusted_data = adjusted_data*np.exp(-1.0j*phase0)
        adjusted_data_list.append(adjusted_data)

    delta_t = waveform_set.delta_t
    aligned_waveforms = [pycbc.types.TimeSeries(adjusted_data_list[i], delta_t=delta_t) for i in range(len(adjusted_data_list))]
    
    return WaveformSet(waveforms=aligned_waveforms, parameters=waveform_set.parameters, regular_grid_shape=waveform_set.regular_grid_shape)


################################################################################
#                             Plotting functions                               #
################################################################################

def plot_time_series_list(waveform_list, real=True, imag=False, mag=True, length=None, xlabel=r'$tc^3/GM$', ylabel=r'$(c^2 d /GM) h$', labels=None):
    """Plot list of waveforms.
    """
    # Get time of maximum of first waveform
    max0, max0i = waveform_list[0].abs_max_loc()
    tmax = waveform_list[0].sample_times[max0i]
    
    if length:
        tstart = tmax - length
    else:
        # Can't use start_time because its type is lal.lal.LIGOTimeGPS instead of numpy.float64.
        tstart = waveform_list[0].sample_times[0]
    
    #tend = waveform_list.end_time
    tend = tstart + 1.1*(tmax-tstart)
    
    fig = plt.figure(figsize=(16, 6))
    axes = fig.add_subplot(1, 1, 1)
    
    # Cycle through the colors black, blue, red, green, then repeat.
    color=['k', 'b', 'r', 'g']*(1+len(waveform_list)/4)
    for i in range(len(waveform_list)):
        # Define labels
        if labels:
            label=labels[i]
        else:
            label=str(i)
        # Make the plots
        if real:
            axes.plot(waveform_list[i].sample_times, waveform_list[i].numpy().real, color=color[i], ls='-', lw=1, label=label)
        if imag:
            axes.plot(waveform_list[i].sample_times, waveform_list[i].numpy().imag, color=color[i], ls='--', lw=1)
        if mag:
            axes.plot(waveform_list[i].sample_times, np.abs(waveform_list[i].numpy()), color=color[i], ls='-', lw=3)
    
    axes.set_xlim([tstart, tend])
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)
    
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    
    axes.legend(fontsize=14, loc='upper left')



def compare_two_waveforms(h1, h2, mag=None, length=None,
                          xlabel=r'$tc^3/GM$', ylabel_wave=r'$(c^2 d /GM) h$',
                          ylabel_amp=r'$A_2/A_1 - 1$',
                          ylabel_phase=r'$\Phi_2 - \Phi_1$',
                          labels=[r'$h_1(t)$', r'$h_2(t)$']):
    """Plot amplitude and phase error
    """
    # Get amplitude and phase of each waveform
    # Don't zero out the starting phase
    amp1, phase1 = amp_phase_from_complex(h1, remove_start_phase=False)
    amp2, phase2 = amp_phase_from_complex(h2, remove_start_phase=False)
    
    # Get time of maximum
    max, maxi_1 = h1.abs_max_loc()
    max, maxi_2 = h2.abs_max_loc()
    maxi = min(maxi_1, maxi_2)
    tmax = h1.sample_times[maxi]
    
    if length:
        tstart = tmax - length
    else:
        # Can't use start_time because its type is lal.lal.LIGOTimeGPS instead of numpy.float64.
        tstart = h1.sample_times[0]
    
    tend = tstart + 1.1*(tmax-tstart)
    
    # Plot waveform
    fig = plt.figure(figsize=(16, 6))
    axes = fig.add_subplot(3, 1, 1)
    axes.plot(h1.sample_times, h1.numpy().real, color='b', ls='-', lw=1, label=labels[0])
    axes.plot(h2.sample_times, h2.numpy().real, color='r', ls='--', lw=1, label=labels[1])
    if mag:
        axes.plot(h1.sample_times, np.abs(h1.numpy()), color='b', ls='-', lw=2)
        axes.plot(h2.sample_times, np.abs(h2.numpy()), color='r', ls='-', lw=2)
    axes.set_xlim([tstart, tend])
    axes.set_ylabel(ylabel_wave, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.legend(fontsize=14, loc='upper left')
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    
    # Plot relative amplitude error
    axes = fig.add_subplot(3, 1, 2)
    axes.plot(amp1.sample_times[:maxi], amp2.numpy()[:maxi]/amp1.numpy()[:maxi]-1, color='b', ls='-', lw=1)
    axes.plot([tstart, tend], [0.0, 0.0], color='k', ls=':', lw=1)
    axes.set_xlim([tstart, tend])
    axes.set_ylabel(ylabel_amp, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    
    # Plot phase error
    axes = fig.add_subplot(3, 1, 3)
    axes.plot(phase1.sample_times[:maxi], phase2.numpy()[:maxi] - phase1.numpy()[:maxi], color='b', ls='-', lw=1)
    axes.plot([tstart, tend], [0.0, 0.0], color='k', ls=':', lw=1)
    axes.set_xlim([tstart, tend])
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel_phase, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    
    subplots_adjust(hspace=0.09)

