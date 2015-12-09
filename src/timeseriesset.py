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
Classes for storing a set of TimeSeries waveforms.
"""

import numpy as np
import h5py
import pycbc.types # TimeSeries
import timeseriesutils as tsutils


################################################################################
#                        HDF5TimeSeriesSet class                               #
################################################################################

class HDF5TimeSeriesSet:
    """Methods for reading and writing TimeSeries waveforms from an hdf5 file
    that contains a list of TimeSeries data.
    
    Attributes
    ----------
    ts_file : h5py.File object
        Pointer to the hdf5 file.
    regular_grid_shape : List of ints
        Shape of the grid.
    param_names : List of strings
        The names of the waveform parameters.
    """
    
    def __init__(self, filename, param_names, regular_grid_shape=None):
        """
        Get a list of waveforms and data associated with those waveforms.
        
        Parameters
        ----------
        filename : string
            Name of the hdf5 file to store the TimeSeriesSet
        param_names: List of strings
            The names of the waveform parameters.
        regular_grid_shape : List like
            Shape of the grid if the waveforms are constructed from a regular grid of parameters.
            Ex. For a list of 125 waveforms constructed by varying 3 parameters over 5 values each,
            the shape would be [5, 5, 5].
        """
        # Read/write if file exists, create otherwise
        self.ts_file = h5py.File(filename, 'a', libver='latest')
        
        self.param_names = param_names
        
        self.regular_grid_shape = regular_grid_shape
        # Store regular_grid_shape in file if it doesn't already exist
        if 'shape' not in self.ts_file.keys():
            if regular_grid_shape is not None:
                self.ts_file['shape'] = regular_grid_shape
            else:
                self.ts_file['shape'] = [0]

    def close(self):
        """Close the hdf5 file.
        """
        # Do you need to also flush() the file from RAM?
        self.ts_file.close()
    
    def add_waveform(self, waveform, parameters, index, amp=None, phase=None):
        """Add a waveform to the hdf5 file.
        
        Parameters
        ----------
        waveform : TimeSeries
        parameters : Parameters of the waveform
        index : int
            For index i, the waveform will be stored under the group 'wave_i'.
        amp : TimeSeries
            The amplitude.
        phase : TimeSeries
            The phase.
        """
        groupname = 'wave_'+str(index)
        wave = self.ts_file.create_group(groupname)
        
        wave['param_names'] = self.param_names
        wave['parameters'] = parameters
        wave['delta_t'] = waveform.delta_t
        # Explicitly cast lal.lal.LIGOTimeGPS to float:
        wave['epoch'] = float(waveform.start_time)
        wave['timeseries'] = waveform.numpy()
        # Calculate amplitude and phase if one or both are not provided
        if not amp or not phase:
            amp, phase = tsutils.amp_phase_from_complex(waveform)
        wave['amplitude'] = amp.numpy()
        wave['phase'] = phase.numpy()
    
    def overwrite_waveform(self, waveform, parameters, index, amp=None, phase=None):
        """Delete a waveform from the hdf5 file, and write over it with a new waveform.
        The waveform can have a different length.
        
        Parameters
        ----------
        waveform : TimeSeries
        parameters : Parameters of the waveform
        index : int
            For index i, the waveform will be stored under the group 'wave_i'.
        amp : TimeSeries
            The amplitude.
        phase : TimeSeries
            The phase.
        """
        # Delete waveform
        groupname = 'wave_'+str(index)
        del self.ts_file[groupname]
        # Add new waveform with same group name
        self.add_waveform(waveform, parameters, index, amp=amp, phase=phase)
    
    def add_waveform_list(self, waveform_list, parameters_list, amp_list=None, phase_list=None):
        """Add a list of N waveforms to the hdf5 file.
        Waveforms will be stored under the groups 'wave_0'--'wave_N.
        
        Parameters
        ----------
        waveform_list : List of TimeSeries
        parameters_list : List of parameters of the waveform
        amp_list : List of TimeSeries
            The amplitudes.
        phase_list : List of TimeSeries
            The phases.
        """
        Nwaves = len(waveform_list)
        for i in range(Nwaves):
            # Calculate amplitude and phase if one or both are not provided
            if not amp_list or not phase_list:
                self.add_waveform(waveform_list[i], parameters_list[i], i, amp=None, phase=None)
            else:
                self.add_waveform(waveform_list[i], parameters_list[i], i, amp=amp_list[i], phase=phase_list[i])

    def get_waveform_data(self, index, data_set='timeseries'):
        """Load a single complex TimeSeries waveform from the HDF5 file.
            
        Parameters
        ----------
        index : int
            Index of the file you want.
        data_set : str, {'timeseries', 'amplitude', 'phase', parameters, param_names}
            The data to extract for the waveform.
    
        Returns
        -------
        TimeSeries for {'timeseries', 'amplitude', 'phase'}
        array of floats for 'parameters'
        array of strings for 'param_names'
        """
        # Get the waveform group
        groupname = 'wave_'+str(index)
        wave = self.ts_file[groupname]
        # Extract the data
        delta_t = wave['delta_t'][()]
        epoch = wave['epoch'][()]
        if data_set == 'timeseries':
            data = wave['timeseries'][:]
            return pycbc.waveform.TimeSeries(data, delta_t=delta_t, epoch=epoch)
        elif data_set == 'amplitude':
            data = wave['amplitude'][:]
            return pycbc.waveform.TimeSeries(data, delta_t=delta_t, epoch=epoch)
        elif data_set == 'phase':
            data = wave['phase'][:]
            return pycbc.waveform.TimeSeries(data, delta_t=delta_t, epoch=epoch)
        elif data_set == 'parameters':
            return wave['parameters'][:]
        elif data_set == 'param_names':
            return wave['param_names'][:]
        else:
            raise Exception, 'Valid data_set options are ' \
            '{timeseries, amplitude, phase, parameters, param_names}. ' \
            'You chose '+str(data_set)

    def get_parameters(self):
        """Get a list of the waveform parameters.
        
        Returns
        -------
        parameters : 2d list
            List of waveform parameters.
        """
        # TODO: One of the keys is 'shape' so subtract one. This is ugly.
        Nwaves = len(self.ts_file.keys())-1
        return [list(self.get_waveform_data(i, data_set='parameters')) for i in range(Nwaves)]

    def get_parameter_grid(self):
        """
        Get the parameters in a grid of dimension Nparams.
        
        Returns
        -------
        parameter_grid : (N+1)d array
            Grid of parameters for N dimensional parameter space.
        """
        if self.regular_grid_shape:
            Nparams = len(self.get_waveform_data(0, data_set='parameters'))
            parameters = self.get_parameters()
            return np.array(parameters).reshape(tuple(self.regular_grid_shape+[Nparams]))
        else:
            raise Exception, 'regular_grid_shape not provided.'


def load_hdf5_time_series_set(filename):
    """Create an HDF5TimeSeriesSet object from a file.
    """
    
    # Crazy way of finding out the param_names and regular_grid_shape
    ts_file = h5py.File(filename, 'a', libver='latest')
    
    param_names = list(ts_file['wave_0/param_names'][:])
    
    if ts_file['shape'][0] == 0:
        regular_grid_shape = None
    else:
        regular_grid_shape = list(ts_file['shape'][:])
    
    ts_file.close()
    
    return tsset.HDF5TimeSeriesSet(filename, param_names, regular_grid_shape=regular_grid_shape)


#################################################################################
##                          TimeSeriesSet class                                 #
#################################################################################
#
#class TimeSeriesSet:
#    
#    def __init__(self, filename=None, waveforms=None, parameters=None, regular_grid_shape=None):
#        """
#        Get a list of waveforms and data associated with those waveforms.
#        
#        Parameters
#        ----------
#        filename : string
#            Name of the file you want to load to. This must be constructed from the function save_waveform_list.
#        waveforms : List of pycbc TimeSeries
#            The list of waveforms
#        parameters : 2-d array like.
#            List or array of the parameters associated with each waveform.
#        
#        regular_grid_shape : List like
#            If your waveforms were constructed from a regular grid, this is the shape of that grid.
#            Ex. For a list of 125 waveforms constructed by varying 3 parameters over 5 values each, the shape would be [5, 5, 5]
#        """
#        
#        if filename:
#            # Load waveform set from file if a file was provided
#            self.regular_grid_shape, self.parameters, self.delta_t, self.start_times, waveform_data_matrix = np.load(filename)
#            self.waveforms = [pycbc.types.TimeSeries(waveform_data_matrix[i], delta_t=self.delta_t, epoch=self.start_times[i]) for i in range(len(waveform_data_matrix))]
#        else:
#            # Get waveform set from a list already created
#            self.waveforms = waveforms
#            self.parameters = parameters
#            self.regular_grid_shape = regular_grid_shape
#            self.delta_t = waveforms[0].delta_t
#            self.start_times = [waveforms[i].start_time for i in range(len(self.waveforms))]
#        
#        self.amp = None
#        self.phase = None
#        self.hp = None
#        self.hc = None
#    
#    # !!!!!!!! Iterating over the index j currently dominates computing time. Why can't you just use the .numpy() method? !!!!!!!!
#    def save(self, outfile):
#        """
#        Saves the waveforms and the data associated with them.
#        
#        Parameters
#        ----------
#        outfile : string
#            Name of the file you want to save to. should be 'something.npy'.
#        """
#        
#        # Generate the data matrix for the N time series
#        # The waveforms can have unequal sizes
#        waveform_data_matrix = [[self.waveforms[i].numpy()[j] for j in range(len(self.waveforms[i]))] for i in range(len(self.waveforms))]
#        
#        # All waveforms must have the same self.delta_t
#        bigarray = [self.regular_grid_shape, self.parameters, self.delta_t, self.start_times, waveform_data_matrix]
#        
#        # Save the file
#        np.save(outfile, bigarray)
#    
#    def calculate_amp_phase_of_set(self):
#        """
#        Construct two real TimeSeries for the amplitude and phase of the complex waveforms.
#        """
#        
#        # Number of waveforms
#        Nwave = len(self.waveforms)
#        
#        # Construct lists of amplitude and phase TimeSeries of the waveforms
#        self.amp = []
#        self.phase = []
#        for i in range(Nwave):
#            if i%100==0: print i,
#            ampi, phasei = tsutils.amp_phase_from_complex(self.waveforms[i])
#            self.amp.append(ampi)
#            self.phase.append(phasei)
#    
#    def calculate_polarizations_of_set(self):
#        """
#        Construct two real TimeSeries for hp and hc from the complex waveforms.
#        """
#        
#        # Number of waveforms
#        Nwave = len(self.waveforms)
#        
#        # Construct lists of amplitude and phase TimeSeries of the waveforms
#        self.hp = []
#        self.hc = []
#        for i in range(Nwave):
#            hpi, hci = complex_to_polarizations(self.waveforms[i])
#            self.hp.append(hpi)
#            self.hc.append(hci)
#
#    def get_parameter_grid(self):
#        """
#        Get the parameters in a grid of dimension Nparams.
#        """
#        if self.regular_grid_shape:
#            Nparams = len(self.parameters[0])
#            return np.array(self.parameters).reshape(tuple(self.regular_grid_shape+[Nparams]))
#        else:
#            raise Exception, 'regular_grid_shape not provided.'
#
#
#
#################################################################################
##                  Functions that use the TimeSeriesSet class                  #
#################################################################################
#
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##!!!!!!! this currently only works for 3-dimensions !!!!!!!!
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#def downsampled_waveform_set_on_grid(ws, d_param_indices):
#    """
#    Parameters
#    ----------
#    ws : WaveformSet
#        The waveform set with a regular_grid_shape.
#    """
#    #!!!!!!! this currently only works for 3-dimensions!!!!!!!!
#    dx, dy, dz = d_param_indices
#    
#    # Put indices of all the waveforms on a grid
#    shape = ws.regular_grid_shape
#    index_grid = np.array(range(len(ws.waveforms))).reshape(shape)
#    index_grid_down = index_grid[::dx, ::dy, ::dz]
#    index_list_down = index_grid_down.flatten()
#    
#    # Construct your new downsampled waveformset
#    waveform_list = [ws.waveforms[i] for i in index_list_down]
#    param_list = [ws.parameters[i] for i in index_list_down]
#    shape = list(index_grid_down.shape)
#    ws_down = WaveformSet(waveforms=waveform_list, parameters=param_list, regular_grid_shape=shape)
#    
#    if ws.amp is not None:
#        ws_down.amp = [ws.amp[i] for i in index_list_down]
#    if ws.phase is not None:
#        ws_down.phase = [ws.phase[i] for i in index_list_down]
#    if ws.hp is not None:
#        ws_down.hp = [ws.hp[i] for i in index_list_down]
#    if ws.hc is not None:
#        ws_down.hc = [ws.hc[i] for i in index_list_down]
#    
#    return ws_down
