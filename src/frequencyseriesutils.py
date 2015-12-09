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
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.pylab import subplots_adjust
import pycbc.types # FrequencySeries

################################################################################
#         Functions to convert between (complex h), (amp, phase).              #
################################################################################


def amplitude_from_frequency_series(htilde):
    """
        Return the amplitude.
        
        Parameters
        ----------
        htilde : FrequencySeries
        A complex PyCBC FrequencySeries.
        
        Returns
        -------
        phase : FrequencySeries
        A real PyCBC FrequencySeries containing the amplitude.
        """
    
    # Calculate the amplitude for each point.
    amp = np.abs(htilde.numpy())
    
    # Return the amplitude as a FrequencySeries
    return pycbc.types.FrequencySeries(amp, delta_f=htilde.delta_f, epoch=htilde.epoch)


def phase_from_frequency_series(htilde, remove_start_phase=True):
    """
        Return the unwrapped phase.
        
        Parameters
        ----------
        htilde : FrequencySeries
        A complex PyCBC FrequencySeries.
        
        Returns
        -------
        phase : FrequencySeries
        A real PyCBC FrequencySeries containing the unwrapped phase.
        """
    
    # Calculate the angle (between -pi and pi) for each point.
    phase_wrapped = np.angle(htilde.numpy())
    
    # Unwrap the phase
    phase = np.unwrap(phase_wrapped)
    
    # Shift the phase to be 0.0 at the first data point
    if remove_start_phase:
        phase += -phase[0]
    
    # Return the phase as a FrequencySeries
    return pycbc.types.FrequencySeries(phase, delta_f=htilde.delta_f, epoch=htilde.epoch)



################################################################################
#                          Linear algebra functions                            #
################################################################################

def check_frequency_series_consistency(h1, h2):
    """Check that h1 and h2 have the same (1) length, (2) delta_f.
        """
    if len(h1) != len(h2):
        raise Exception, 'len(h1)='+str(len(h1))+' and len(h1)='+str(len(h1))+' are not the same.'
    if h1.delta_f != h2.delta_f:
        raise Exception, 'h1.delta_f='+str(h1.delta_f)+' and h2.delta_f='+str(h1.delta_f)+' are not the same.'


def add_frequency_series(h1, h2):
    """Evaluate h1+h2.
        Assumes h1 and h2 have the same (1) length, (2) delta_f.
        """
    check_frequency_series_consistency(h1, h2)
    return pycbc.types.FrequencySeries(h1+h2, delta_f=h1.delta_f)


def subtract_frequency_series(h1, h2):
    """Evaluate h1-h2.
        Assumes h1 and h2 have the same (1) length, (2) delta_f.
        """
    check_frequency_series_consistency(h1, h2)
    return pycbc.types.FrequencySeries(h1-h2, delta_f=h1.delta_f)


def scalar_multiply_frequency_series(alpha, h):
    """Multiply a waveform h by a float alpha.
        Assumes h1 and h2 have the same (1) length, (2) delta_f.
        """
    return pycbc.types.FrequencySeries(alpha*h, delta_f=h.delta_f)


def inner_product_frequency_series(h1, h2):
    """Evaluate the inner product < h1, h2 > = int_fL^fH df h1*(f) h2(f).
        Assumes h1 and h2 have the same (1) length, (2) delta_f.
        """
    check_frequency_series_consistency(h1, h2)
    integrand = np.sum( np.array(h1.numpy().tolist()).conj() * np.array(h2.numpy().tolist()) )
    inner = h1.delta_f * integrand
    return inner


################################################################################
#                             Plotting functions                               #
################################################################################

def plot_frequency_series_list(waveform_list, real=True, imag=False, mag=True, remove_start_phase=False,
                               xlabel=r'$GMf/c^3$', ylabel=r'$\tilde h$', labels=None, xlim=None, ymin=None):
    """Plot list of FrequencySeries waveforms.
    """
    
    # Cycle through the colors black, blue, red, green, then repeat.
    color=['k', 'b', 'r', 'g']*(1+len(waveform_list)/4)
    
    fig = plt.figure(figsize=(16, 12))
    
    
    axes = fig.add_subplot(3, 1, 1)
    for i in range(len(waveform_list)):
        # Define labels
        if labels:
            label=labels[i]
        else:
            label=str(i)
        # Make the plots
        if real:
            axes.semilogx(waveform_list[i].sample_frequencies, waveform_list[i].numpy().real, color=color[i], ls='-', lw=1, label=label)
        if imag:
            axes.semilogx(waveform_list[i].sample_frequencies, waveform_list[i].numpy().imag, color=color[i], ls='--', lw=1)
        if mag:
            axes.semilogx(waveform_list[i].sample_frequencies, np.abs(waveform_list[i].numpy()), color=color[i], ls='-', lw=3)
    
    if xlim: axes.set_xlim(xlim)
    #axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)
    
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    axes.legend(fontsize=14, loc='upper right')


    axes = fig.add_subplot(3, 1, 2)
    for i in range(len(waveform_list)):
        hamp = amplitude_from_frequency_series(waveform_list[i])
        axes.loglog(hamp.sample_frequencies, hamp.numpy(), color=color[i], ls='-', lw=3)

    if xlim: axes.set_xlim(xlim)
    if ymin: axes.set_ylim([ymin, 1.1*max(hamp)])
    #axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel('Amplitude', fontsize=16)
    
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    axes.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    
    axes = fig.add_subplot(3, 1, 3)
    for i in range(len(waveform_list)):
        hphase = phase_from_frequency_series(waveform_list[i], remove_start_phase=remove_start_phase)
        axes.semilogx(hphase.sample_frequencies, hphase.numpy(), color=color[i], ls='-', lw=3)

    if xlim: axes.set_xlim(xlim)
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel('Phase', fontsize=16)
    
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)
    
    subplots_adjust(hspace=0.05)



class FrequencySeriesSet:
    
    def __init__(self, filename=None, waveforms=None, parameters=None, regular_grid_shape=None):
        """
            Get a list of waveforms and data associated with those waveforms.
            
            Parameters
            ----------
            filename : string
            Name of the file you want to load to. This must be constructed from the function save_waveform_list.
            waveforms : List of pycbc TimeSeries
            The list of waveforms
            parameters : 2-d array like.
            List or array of the parameters associated with each waveform.
            
            regular_grid_shape : List like
            If your waveforms were constructed from a regular grid, this is the shape of that grid.
            Ex. For a list of 125 waveforms constructed by varying 3 parameters over 5 values each, the shape would be [5, 5, 5]
            """
        
        if filename:
            # Load waveform set from file if a file was provided
            self.regular_grid_shape, self.parameters, self.delta_t, self.start_times, waveform_data_matrix = np.load(filename)
            self.waveforms = [pycbc.types.TimeSeries(waveform_data_matrix[i], delta_t=self.delta_t, epoch=self.start_times[i]) for i in range(len(waveform_data_matrix))]
        else:
            # Get waveform set from a list already created
            self.waveforms = waveforms
            self.parameters = parameters
            self.regular_grid_shape = regular_grid_shape
            self.delta_t = waveforms[0].delta_t
            self.start_times = [waveforms[i].start_time for i in range(len(self.waveforms))]
        
        self.amp = None
        self.phase = None
        self.hp = None
        self.hc = None
    
    # !!!!!!!! Iterating over the index j currently dominates computing time. Why can't you just use the .numpy() method? !!!!!!!!
    def save(self, outfile):
        """
            Saves the waveforms and the data associated with them.
            
            Parameters
            ----------
            outfile : string
            Name of the file you want to save to. should be 'something.npy'.
            """
        
        # Generate the data matrix for the N time series
        # The waveforms can have unequal sizes
        waveform_data_matrix = [[self.waveforms[i].numpy()[j] for j in range(len(self.waveforms[i]))] for i in range(len(self.waveforms))]
        
        # All waveforms must have the same self.delta_t
        bigarray = [self.regular_grid_shape, self.parameters, self.delta_t, self.start_times, waveform_data_matrix]
        
        # Save the file
        np.save(outfile, bigarray)
    
    def calculate_amp_phase_of_set(self):
        """
            Construct two real TimeSeries for the amplitude and phase of the complex waveforms.
            """
        
        # Number of waveforms
        Nwave = len(self.waveforms)
        
        # Construct lists of amplitude and phase TimeSeries of the waveforms
        self.amp = []
        self.phase = []
        for i in range(Nwave):
            if i%100==0: print i,
            ampi, phasei = amp_phase_from_complex(self.waveforms[i])
            self.amp.append(ampi)
            self.phase.append(phasei)
    
    def calculate_polarizations_of_set(self):
        """
            Construct two real TimeSeries for hp and hc from the complex waveforms.
            """
        
        # Number of waveforms
        Nwave = len(self.waveforms)
        
        # Construct lists of amplitude and phase TimeSeries of the waveforms
        self.hp = []
        self.hc = []
        for i in range(Nwave):
            hpi, hci = complex_to_polarizations(self.waveforms[i])
            self.hp.append(hpi)
            self.hc.append(hci)

def get_parameter_grid(self):
    """
        Get the parameters in a grid of dimension Nparams.
        """
            
            if self.regular_grid_shape:
            Nparams = len(self.parameters[0])
            return np.array(self.parameters).reshape(tuple(self.regular_grid_shape+[Nparams]))
                else:
                    raise Exception, 'regular_grid_shape not provided.'
