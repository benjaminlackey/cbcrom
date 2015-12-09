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
import h5py

import scipy.interpolate
import scipy.optimize

import glob
import operator

import timedomainwaveform as tdwave
#import timeseriesutils as tsutils
#import timeseriesset as tsset
#import taylort4trainingset as taylort4
#import diagnostics
#import pycbc.types # TimeSeries



################################################################################
#               Read in Sebastiano's hdf5 tidal EOB files                      #
################################################################################
class File:
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, path, mode='r'):
        
        # Check file extension
        ext = path.split('.')[-1]
        if ext not in ['hdf5', 'h5']:
            raise Exception, "Expecting hdf5 or h5 format."
        
        self.path = path
        self.mode_options = ['r', 'w', 'w+', 'a']
        
        pass
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def open(self, path, mode='r'):
        if mode in self.mode_options:
            try:
                self.file = h5py.File(path, mode)
                self.flag = 1
                self.keys = self.file.keys()  # Get all keys (e.g., variable names)
            except IOError:
                print "Could not open file."
                self.flag = 0
        else:
            raise Exception, "File action not recognized."

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def isopen(self):
        if self.flag == 1:
            return True
        else:
            return False
            
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def close(self):
        self.file.close()
        self.flag = 0
        pass


def time_amp_phase_from_filename(filename):
    
    waveform = File(filename, mode='r')
    waveform.open(filename, mode='r')
    
    # 'rwz' stands for regge-wheeler-zerilli ?
    time = waveform.file['time/time'][:]
    amp = waveform.file['rwz/amplitude/Al2m2'][:]
    phase = waveform.file['rwz/phase/phil2m2'][:]
    
    return time, amp, phase


################################################################################
#       Construct the training set as a HDF5TimeDomainWaveformSet              #
################################################################################

def time_at_max_amp(time, amp):
    """Find the time corresponding to the maximum amplitude.
    This function interpolates between data points using order 2 polynomial,
    before finding the maximum.
    """
    
    # Find index of maximum amplitude
    imax = np.argmax(amp)
    tmax = time[imax]
    
    # Just interpolate a small number of points surrounding imax
    #!!!!!!!!! This could cause errors if you don't have 3 points surrounding imax !!!!!!!!!!!
    tlist = time[imax-3:imax+3]
    alist = amp[imax-3:imax+3]
    
    # Use 2nd order interpolation
    # because it's safe and gives pretty much the same results as higher order interpolation.
    negampoft = scipy.interpolate.UnivariateSpline(tlist, -1.0*alist, k=2, s=0)
    
    # Find the minimum, starting the search at tmax
    result = scipy.optimize.minimize(negampoft, tmax)
    
    return result.x[0]


def get_eob_training_set(training_set_dir, ts_filename, memb_size=2**31-1, regular_grid=False,
                         param_names=['q', 'LambdaA', 'LambdaB'], samples_per_cycle=100.0):
    """
    Parameters
    ----------
    training_set_dir : str
        Name of the directory containing the waveform files.
    ts_filename : str
        Name of the hdf5 file to store the training set
    delta_t : float
        Time sample spacing.
    regular_grid : {bool, False}
        True if parameters are on a regular grid with rectangular bounds,
        False otherwise.
    
    Returns
    -------
    waveform_set : HDF5TimeDomainWaveformSet
        The equal-length, aligned training set.
    """

    # Get filenames.
    # Extract the parameters from the filenames.
    # Sort the list by the parameters.

    filenames = glob.glob(training_set_dir+'/*.h5')

    # Number of waveforms
    Nwave = len(filenames)

    file_and_params = []
    for filename in filenames:
        params = map(float, filename.split('/')[-1].split('.')[0].replace('p', '.').split('_')[-3:])
        file_and_params.append([filename]+params)

    # Convert mass ratio bigq to smallq
    for i in range(Nwave):
        bigq = file_and_params[i][1]
        file_and_params[i][1] = 1.0/bigq

    # Sort the waveforms by their parameter values
    file_and_params_sorted = sorted(file_and_params, key=operator.itemgetter(1, 2, 3))
    param_list = [file_and_params_sorted[i][1:] for i in range(Nwave)]

    if regular_grid:
        # Make a regular grid of parameters
        params_array = np.array(param_list)
        # Set gives a dictionary of unique elements in a list
        Nparams = len(params_array[0,:])
        regular_grid_shape = [len(set(params_array[:,i])) for i in range(Nparams)]
        #params_grid = params_array.reshape(tuple(regular_grid_shape+[Nparams]))
    else:
        regular_grid_shape = None
    
    # Create waveform set object
    waveform_set = tdwave.HDF5TimeDomainWaveformSet(ts_filename, mode='x', memb_size=memb_size)
    waveform_set.set_parameter_names(param_names)
    waveform_set.set_grid_shape(shape=regular_grid_shape)
    
    ########### Get all waveforms, do time shift, clean them up, then place them in a common object. ##########
    tstartlist = []
    tendlist = []
    for i in range(Nwave):
        if i%100==0: print i,
        #print i,
        # Import waveform.
        filename = file_and_params_sorted[i][0]
        time, amp, phase = time_amp_phase_from_filename(filename)
        h = tdwave.TimeDomainWaveform(time, amp, phase)
        # Align at max amplitude.
        tatmax = time_at_max_amp(h.time, h.amp)
        h.time_shift(-tatmax)
        # Resample and remove decreasing phase at end.
        h.resample(samples_per_cycle=samples_per_cycle)
        # Add to hdf5 file.
        tstartlist.append(h.time[0])
        tendlist.append(h.time[-1])
        waveform_set.add_waveform(h, param_list[i], i)
    
    ############# Make sure all waveforms have the same time samples. ############
    # Find common start and end times.
    tstartnew = max(tstartlist)
    tendnew = min(tendlist)
    # Use 0th waveform to determine time samples.
    h0 = waveform_set.get_waveform_data(0)
    h0.resample(t_start=tstartnew, t_end=tendnew, samples_per_cycle=samples_per_cycle)
    time_new = h0.time
    h0.phase_shift(remove_start_phase=True)
    ell = 2
    znorm = np.sqrt((ell+2)*(ell+1)*ell*(ell-1))
    h0.amp *= znorm
    waveform_set.overwrite_waveform(h0, param_list[0], 0)
    # Zero the starting phase, then change normalization convention to go from Zerilli to h_lm.
    for i in range(1, Nwave):
        if i%100==0: print i,
        #print i,
        # Use same time samples for all other waveforms
        h = waveform_set.get_waveform_data(i)
        h.resample(time_new=time_new)
        h.phase_shift(remove_start_phase=True)
        h.amp *= znorm
        waveform_set.overwrite_waveform(h, param_list[i], i)

    return waveform_set


################################################################################
#                                Old junk                                      #
################################################################################

#def resample_time_series_with_interpolation(time, data, delta_t, order=2, t_include=None, t0=None):
#    """
#        Construct an evenly spaced TimeSeries
#        from unevenly sampled lists of time and the corresponding data.
#        
#        This will only contain the last time and data point
#        if time[-1]-time[0] is an integer multiple of delta_t.
#        
#        Parameters
#        ----------
#        time : numpy array
#        data : numpy array
#        delta_t : int
#        The desired sample rate.
#        order : int
#        Order for the interpolating polynomial.
#        t0 : float
#        Time to be reset to 0.0.
#        Defaults to not resetting time.
#        t_include : float
#        The time to include as an exact data point.
#        Defaults to time[0].
#        
#        Returns
#        -------
#        TimeSeries
#        """
#    
#    # Determine the sample times
#    if t_include is None:
#        # Start at first sample
#        sample_times = np.arange(time[0], time[-1], delta_t)
#    else:
#        # Include the time t_include
#        if t_include<time[0] or t_include>time[-1]:
#            raise Exception, 't_include is outside the range of the time array.'
#        Nbelow = np.floor((t_include-time[0])/delta_t)
#        #print Nbelow
#        tstart = t_include-Nbelow*delta_t
#        sample_times = np.arange(tstart, time[-1], delta_t)
#
#    # Do the resampling
#    dataoft = scipy.interpolate.UnivariateSpline(time, data, k=order, s=0)
#    sample_data = dataoft(sample_times)
#
#    # Determine the time shift to apply
#    # epoch is the time of the first data point
#    if t0 is None:
#        epoch = sample_times[0]
#    else:
#        epoch = sample_times[0] - t0
#
#    return pycbc.waveform.TimeSeries(sample_data, delta_t=delta_t, epoch=epoch)
#
#
#def complex_from_resampled_amp_phase_with_interpolation(time, amp, phase, delta_t, order=2, t_include=None, t0=None):
#    """
#        Construct an evenly spaced TimeSeries for the complex waveform h_plus + i*h_cross
#        from unevenly sampled lists of time and the corresponding amplitude and phase.
#        
#        This will only contain the last time and data point
#        if time[-1]-time[0] is an integer multiple of delta_t.
#        
#        Parameters
#        ----------
#        time : numpy array
#        amp : numpy array
#        phase : numpy array
#        delta_t : int
#        The desired sample rate.
#        order : int
#        Order for the interpolating polynomial.
#        
#        Returns
#        -------
#        Complex TimeSeries
#        """
#    
#    amp = resample_time_series_with_interpolation(time, amp, delta_t, order=order, t_include=t_include, t0=t0)
#    phase = resample_time_series_with_interpolation(time, phase, delta_t, order=order, t_include=t_include, t0=t0)
#    #print len(amp), len(phase)
#    
#    return tsutils.complex_from_amp_phase(amp, phase)
#
#
#def tmax_from_time_amp(time, amp):
#    """
#        Find the time corresponding to the maximum amplitude.
#        This function interpolates between data points using order 2 polynomial,
#        before finding the maximum.
#        """
#    
#    # Find index of maximum amplitude
#    imax = np.argmax(amp)
#    tmax = time[imax]
#    
#    # Just interpolate a small number of points surrounding imax
#    #!!!!!!!!! This could cause errors if you don't have 3 points surrounding imax !!!!!!!!!!!
#    tlist = time[imax-3:imax+3]
#    alist = amp[imax-3:imax+3]
#    
#    # Use 2nd order interpolation
#    # because it's safe and gives pretty much the same results as higher order interpolation.
#    negampoft = scipy.interpolate.UnivariateSpline(tlist, -1.0*alist, k=2, s=0)
#    
#    # Find the minimum, starting the search at tmax
#    result = scipy.optimize.minimize(negampoft, tmax)
#    
#    return result.x[0]
#
#
#def shift_waveform_time_to_zero_at_max_amp(time, amp, phase, delta_t, order=2):
#    """
#        Construct an evenly sampled complex waveform,
#        where the maximum amplitude is at t=0.
#        """
#    
#    tmax = tmax_from_time_amp(time, amp)
#    hcomplex = complex_from_resampled_amp_phase_with_interpolation(time, amp, phase, delta_t, order=order, t_include=tmax, t0=tmax)
#    
#    return hcomplex, hcomplex.sample_times[0], hcomplex.sample_times[-1]
#
#
#
#def get_eob_training_set(training_set_dir, ts_filename, delta_t, regular_grid=False, param_names=['eta', 'LambdaA', 'LambdaB']):
#    """
#    Parameters
#    ----------
#    training_set_dir : str
#        Name of the directory containing the waveform files.
#    ts_filename : str
#        Name of the hdf5 file to store the training set
#    delta_t : float
#        Time sample spacing.
#    regular_grid : {bool, False}
#        True if parameters are on a regular grid with rectangular bounds,
#        False otherwise.
#    
#    Returns
#    -------
#    waveform_set : HDF5TimeSeriesSet
#        The equal-length, aligned training set.
#    """
#
#    # Get filenames.
#    # Extract the parameters from the filenames.
#    # Sort the list by the parameters.
#
#    filenames = glob.glob(training_set_dir+'/*.h5')
#
#    # Number of waveforms
#    Nwave = len(filenames)
#
#    file_and_params = []
#    for filename in filenames:
#        params = map(float, filename.split('/')[1].split('.')[0].replace('p', '.').split('_')[-3:])
#        file_and_params.append([filename]+params)
#
#    # Convert mass ratio bigq to eta
#    for i in range(Nwave):
#        bigq = file_and_params[i][1]
#        file_and_params[i][1] = taylort4.eta_of_q(bigq)
#
#    # Sort the waveforms by their parameter values
#    file_and_params_sorted = sorted(file_and_params, key=operator.itemgetter(1, 2, 3))
#    param_list = [file_and_params_sorted[i][1:] for i in range(Nwave)]
#
#    if regular_grid:
#        # Make a regular grid of parameters
#        params_array = np.array(param_list)
#        # Set gives a dictionary of unique elements in a list
#        Nparams = len(params_array[0,:])
#        regular_grid_shape = [len(set(params_array[:,i])) for i in range(Nparams)]
#        #params_grid = params_array.reshape(tuple(regular_grid_shape+[Nparams]))
#    else:
#        regular_grid_shape = None
#
#    # Import the waveforms, align them at max amp, then store them in hdf5 file
#    waveform_set = tsset.HDF5TimeSeriesSet(ts_filename, param_names, regular_grid_shape=regular_grid_shape)
#    tstartlist = []
#    tendlist = []
#    for i in range(Nwave):
#        if i%100==0: print i,
#        filename = file_and_params_sorted[i][0]
#        time, amp, phase = time_amp_phase_from_filename(filename)
#        h, tstart, tend = shift_waveform_time_to_zero_at_max_amp(time, amp, phase, delta_t)
#        waveform_set.add_waveform(h, param_list[i], i, amp=None, phase=None)
#        tstartlist.append(tstart)
#        tendlist.append(tend)
#
#    # Truncate some of the waveforms so they have the same starting and ending times
#    tstartnew = max(tstartlist)
#    tendnew = min(tendlist)
#    for i in range(Nwave):
#        if i%100==0: print i,
#        h = waveform_set.get_waveform_data(i, data_set='timeseries')
#        delta_t = h.delta_t
#        iattstartnew = int(np.rint((tstartnew-h.sample_times[0])/delta_t))
#        iattendnew = int(np.rint((tendnew-h.sample_times[0])/delta_t))
#        data_trunc = h.numpy()[iattstartnew:iattendnew+1]
#
#        # Shift phase to be zero at start of the new truncated waveform hnew
#        # and multiply by Zerilli normalization factor to get the h_lm component
#        ell = 2
#        znorm = np.sqrt((ell+2)*(ell+1)*ell*(ell-1))
#        phase0 = np.angle(data_trunc[0])
#        data_rotate = data_trunc*znorm*np.exp(-1.0j*phase0)
#        hnew = pycbc.waveform.TimeSeries(data_rotate, delta_t=delta_t, epoch=tstartnew)
#        waveform_set.overwrite_waveform(hnew, param_list[i], i, amp=None, phase=None)
#
#    #return waveform_set, tstartlist, tendlist
#    return waveform_set


#def get_eob_training_set(training_set_dir, delta_t, regular_grid=False):
#    """
#    Parameters
#    ----------
#    regular_grid : {bool, False}
#        True if parameters are on a regular grid with rectangular bounds,
#        False otherwise.
#    """
#    
#    # Get filenames.
#    # Extract the parameters from the filenames.
#    # Sort the list by the parameters.
#    
#    filenames = glob.glob(training_set_dir+'/*.h5')
#    
#    # Number of waveforms
#    Nwave = len(filenames)
#    
#    file_and_params = []
#    for filename in filenames:
#        params = map(float, filename.split('/')[1].split('.')[0].replace('p', '.').split('_')[-3:])
#        file_and_params.append([filename]+params)
#    
#    # Convert mass ratio bigq to eta
#    for i in range(Nwave):
#        bigq = file_and_params[i][1]
#        file_and_params[i][1] = taylort4.eta_of_q(bigq)
#    
#    file_and_params_sorted = sorted(file_and_params, key=operator.itemgetter(1, 2, 3))
#    param_list = [file_and_params_sorted[i][1:] for i in range(Nwave)]
#
#    if regular_grid:
#        # Make a regular grid of parameters
#        params_array = np.array(param_list)
#        # Set gives a dictionary of unique elements in a list
#        Nparams = len(params_array[0,:])
#        regular_grid_shape = [len(set(params_array[:,i])) for i in range(Nparams)]
#        #params_grid = params_array.reshape(tuple(regular_grid_shape+[Nparams]))
#    else:
#        regular_grid_shape = None
#
#    # Now actually import the waveforms and align them
#    waveforms = []
#    tstartlist = []
#    tendlist = []
#    for i in range(Nwave):
#        #for i in range(200):
#        if i%100==0: print i,
#        filename = file_and_params_sorted[i][0]
#        time, amp, phase = time_amp_phase_from_filename(filename)
#        h, tstart, tend = shift_waveform_time_to_zero_at_max_amp(time, amp, phase, delta_t)
#        waveforms.append(h)
#        tstartlist.append(tstart)
#        tendlist.append(tend)
#    
#    # resample the waveforms so they have the same starting and ending times
#    tstartnew = max(tstartlist)
#    tendnew = min(tendlist)
#    for i in range(Nwave):
#        #for i in range(200):
#        if i%100==0: print i,
#        
#        # Truncate the waveforms at the beginning and end so they all have the same length
#        h = waveforms[i]
#        delta_t = h.delta_t
#        iattstartnew = int(np.rint((tstartnew-h.sample_times[0])/delta_t))
#        iattendnew = int(np.rint((tendnew-h.sample_times[0])/delta_t))
#        data_trunc = h.numpy()[iattstartnew:iattendnew+1]
#        
#        # Shift phase to be zero at start of the new truncated waveform hnew
#        # and multiply by Zerilli normalization factor to get the h_lm component
#        ell = 2
#        znorm = np.sqrt((ell+2)*(ell+1)*ell*(ell-1))
#        phase0 = np.angle(data_trunc[0])
#        data_rotate = data_trunc*znorm*np.exp(-1.0j*phase0)
#        hnew = pycbc.waveform.TimeSeries(data_rotate, delta_t=delta_t, epoch=tstartnew)
#        waveforms[i] = hnew
#    
#    return tsset.TimeSeriesSet(waveforms=waveforms, parameters=param_list, regular_grid_shape=regular_grid_shape), tstartlist, tendlist
