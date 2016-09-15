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

def truncate_beginning(h, t_trunc):
    """Truncate the first t_trunc of the waveform.
    
    Returns
    ------
    h_trunc : TimeDomainWaveform
    """
    i_trunc = next(i for i in range(len(h.time)-1) if h.time[i]>=t_trunc)
    h_trunc = tdwave.TimeDomainWaveform(h.time[i_trunc:], h.amp[i_trunc:], h.phase[i_trunc:])
    return h_trunc


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


def uniform_in_phase_then_time(h, t_start, t_end, t_transition, samples_per_cycle, dt, order=2):
    """Resample waveform with spacing uniform in phase at the beginning, 
    then transition to uniform in time at the end.
    """
    # Remove points at the end where the phase is decreasing.
    h.remove_decreasing_phase()
    
    # Check that t_start and t_end are in the range of h.time
    if t_start<h.time[0] or t_end>h.time[-1]:
        raise Exception, 'Start and end times must be in the range of h.time.'
        
    # Interpolate time(phase)
    tofphase = scipy.interpolate.UnivariateSpline(h.phase, h.time, k=order, s=0)
    ampoft = scipy.interpolate.UnivariateSpline(h.time, h.amp, k=order, s=0)
    phaseoft = scipy.interpolate.UnivariateSpline(h.time, h.phase, k=order, s=0)
    
    # Uniform phase samples
    phi_start = phaseoft(t_start)
    phi_transition = phaseoft(t_transition)
    #phase = np.arange(phi_start, phi_end, 2.0*np.pi/samples_per_cycle)
    npoints = int(np.ceil((phi_transition - phi_start)*samples_per_cycle/(2.0*np.pi))) + 1
    phase = np.linspace(phi_start, phi_transition, npoints)
    # time samples
    time_a = tofphase(phase)
    time_a[0] = t_start # Make sure this is exactly as requested (not just approximate)
    
    # Uniform time samples
    npoints = int(np.ceil( (t_end-(t_transition+dt))/dt ))
    time_b = np.linspace(t_transition+dt, t_end, npoints)
    
    h.time = np.concatenate((time_a, time_b))
    h.amp = ampoft(h.time)
    h.phase = phaseoft(h.time)


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


def get_eob_training_set_efficient(training_set_dir, ts_filename, memb_size=2**31-1, regular_grid=False,
                         param_names=['q', 'LambdaA', 'LambdaB'], 
                         t_transition=-1000.0, samples_per_cycle=100.0, dt=0.1, t_trunc=1.0e5):
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
        #if i%100==0: print i,
        print i,
        # Import waveform.
        filename = file_and_params_sorted[i][0]
        time, amp, phase = time_amp_phase_from_filename(filename)
        h = tdwave.TimeDomainWaveform(time, amp, phase)
        # Truncate the first t_trunc of the waveform
        h = truncate_beginning(h, t_trunc)
        # Align at max amplitude.
        tatmax = time_at_max_amp(h.time, h.amp)
        h.time_shift(-tatmax)
        # Remove decreasing phase
        h.remove_decreasing_phase()
        uniform_in_phase_then_time(h, h.time[0], h.time[-1], t_transition, samples_per_cycle, dt, order=2)
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
    uniform_in_phase_then_time(h0, tstartnew, tendnew, t_transition, samples_per_cycle, dt, order=2)
    #h0.resample(t_start=tstartnew, t_end=tendnew, samples_per_cycle=samples_per_cycle)
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
        h.resample_at_times(time_new=time_new)
        h.phase_shift(remove_start_phase=True)
        h.amp *= znorm
        waveform_set.overwrite_waveform(h, param_list[i], i)

    return waveform_set


