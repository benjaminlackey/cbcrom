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

# Modules for generating waveforms
import lalsimulation # Get waveform functions
import lal # Get constants
import pycbc.types # TimeSeries
import pycbc.waveform # Waveforms

# Constants
MPC_SI = 1.0e6 * lal.PC_SI

import timeseriesutils as tsutils


################################################################################
##################  Common functions for binary systems       ##################
################################################################################


# Mchirp and eta do not depend on which mass (m1 or m2) is greater.
# Going backwards requires a choice for which mass is greater.
def mchirp_of_m1_m2(m1, m2):
    return (m1*m2)**(3.0/5.0) / (m1+m2)**(1.0/5.0)

def eta_of_m1_m2(m1, m2):
    return (m1*m2) / (m1+m2)**2.0

def eta_of_q(q):
    """
        Takes either big Q=m_1/m_2 or little q=m_2/m_1 and returns
        symmetric mass ratio eta.
        """
    return q / (1.0 + q)**2

def big_and_small_q_of_eta(eta):
    bigq = (1.0-2.0*eta + np.sqrt(1.0-4.0*eta))/(2.0*eta)
    smallq = (1.0-2.0*eta - np.sqrt(1.0-4.0*eta))/(2.0*eta)
    
    return bigq, smallq

def m1_of_mchirp_eta(mchirp, eta):
    """
        m1 is always the more massive star (the primary)
        """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 + np.sqrt(1.0-4.0*eta))


def m2_of_mchirp_eta(mchirp, eta):
    """
        m2 is always the less massive star (the secondary)
        """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 - np.sqrt(1.0-4.0*eta))

def lamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """
        $\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
        Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
        Lambda_2 is for the secondary star m_2.
        """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lam1+lam2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lam1-lam2))

def deltalamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """
        This is the definition found in Les Wade's paper.
        Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper.
        $\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
        Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
        Lambda_2 is for the secondary star m_2.
        """
    return (1.0/2.0)*(
                      np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(lam1+lam2)
                      + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(lam1-lam2)
                      )

def lam1_lam2_of_pe_params(eta, lamt, dlamt):
    """
        lam1 is for the the primary mass m_1.
        lam2 is for the the secondary mass m_2.
        m_1 >= m2.
        """
    
    a = (8.0/13.0)*(1.0+7.0*eta-31.0*eta**2)
    b = (8.0/13.0)*np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)
    c = (1.0/2.0)*np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)
    d = (1.0/2.0)*(1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)
    
    den = (a+b)*(c-d) - (a-b)*(c+d)
    
    lam1 = ( (c-d)*lamt - (a-b)*dlamt )/den
    lam2 = (-(c+d)*lamt + (a+b)*dlamt )/den
    
    return lam1, lam2


################################################################################
#  Functions for downsampling
################################################################################

#def downsample_time_series(waveform, di):
#    """
#        Downsample a waveform by skipping every di data points.
#        
#        Parameters
#        ----------
#        waveform : TimeSeries
#        di : int
#        Spacing. 1 returns the same waveform, 2 skips every other point, etc.
#        
#        Returns
#        -------
#        downsampled_waveform : TimeSeries
#        The downsampled time series
#        """
#    
#    downsampled_data = waveform.numpy()[::di]
#    delta_t_new = waveform.delta_t * di
#    
#    return pycbc.types.TimeSeries(downsampled_data, delta_t=delta_t_new)
#
#
###############################
##!!!!!!!!!!!! This might be failing. Is the maximum amplitude always the same as the last data point? !!!!!!!!!!!
## Maybe calculate both to check if you always get the same frequency.
###############################
#def downsample_time_series_from_end(waveform, di):
#    """
#        Downsample a waveform by skipping every di data points, making sure you use the last point and truncate at the beginning.
#        
#        Parameters
#        ----------
#        waveform : TimeSeries
#        di : int
#        Spacing. 1 returns the same waveform, 2 skips every other point, etc.
#        
#        Returns
#        -------
#        downsampled_waveform : TimeSeries
#        The downsampled time series
#        """
#    
#    # Pick the starting index so the last resampled point is the last data point
#    # This will truncate the elements istart-1 and below
#    istart = (len(waveform)-1)%di
#    downsampled_data = waveform.numpy()[istart::di]
#    delta_t_new = waveform.delta_t * di
#    
#    return pycbc.types.TimeSeries(downsampled_data, delta_t=delta_t_new)


def downsample_time_series_include_max_amp(waveform, di):
    """
        Downsample a waveform by skipping every di data points, making sure you use the last point and truncate at the beginning.
        
        Parameters
        ----------
        waveform : TimeSeries
        di : int
        Spacing. 1 returns the same waveform, 2 skips every other point, etc.
        
        Returns
        -------
        downsampled_waveform : TimeSeries
        The downsampled time series
        """
    
    # Pick the starting index so the maximum amplitude will be one of the resampled points
    # This will truncate the elements istart-1 and below
    maxamp, imax = waveform.abs_max_loc()
    istart = imax%di
    downsampled_data = waveform.numpy()[istart::di]
    delta_t_new = waveform.delta_t * di
    
    return pycbc.types.TimeSeries(downsampled_data, delta_t=delta_t_new)


################################################################################
#  Functions for generating the training_set with TaylorT4 waveforms
################################################################################


def pn_waveform(m1, m2, lambda1, lambda2, f_lower, f_samp, pn_type='TaylorT4'):
    """
    Wrapper for pycbc waveform generator.
    """
    
    # Reference frequency at which phase is set to 0
    f_ref = f_lower
    delta_t = 1.0/f_samp
    dist = lal.G_SI*(m1+m2)*lal.MSUN_SI/(lal.C_SI**2*MPC_SI)
    #print dist, delta_t
    
    hp, hc = pycbc.waveform.get_td_waveform(approximant=pn_type, mass1=m1, mass2=m2, lambda1=lambda1, lambda2=lambda2,
                                            distance=dist, delta_t=delta_t, f_lower=f_lower, f_ref=f_ref,
                                            phase_order=7, tidal_order=12, amplitude_order=6)

    # Rescale time to dimensionless units
    delta_tdim = delta_t*lal.C_SI**3/(lal.G_SI*(m1+m2)*lal.MSUN_SI)

    # h = hp + 1j*hc
    return pycbc.types.TimeSeries(hp.numpy()+1.0j*hc.numpy(), delta_t=delta_tdim)


def pn_waveform_of_pe_params(f1010_low, f1010_samp, eta, lamt, dlamt, pn_type='TaylorT4'):
    """"
        Generate a post-Newtonian waveform using the parameters used in parameter estimation.
        The length will be the same as a 1.0--1.0Msun binary starting at f1010_low
        Parameters
        ----------
        f1010_low : float
        Starting Frequency (Hz) of a 1.0--1.0Msun binary.
        f1010_samp : float
        Sampling frequency
        """
    
    # Fix smallest mass at 1.0Msun
    m2 = 1.0
    Q, q = big_and_small_q_of_eta(eta)
    m1 = Q*m2
    
    # Scale with total mass
    # Must start at lower frequency if using (total mass) > (1.0Msun + 1.0Msun)
    # so that length of all the waveforms are a long as the length for a 1.0--1.0Msun binary
    f_low = f1010_low*(2.0*m2)/(m1+m2)
    f_samp = f1010_samp*(2.0*m2)/(m1+m2)
    
    lam1, lam2 = lam1_lam2_of_pe_params(eta, lamt, dlamt)
    
    #print m1, m2, lam1, lam2, f_low, f_samp
    return pn_waveform(m1, m2, lam1, lam2, f_low, f_samp, pn_type)
#return pn_waveform(m1, m2, lam1, lam2, f_low, f_samp, pn_type)


def waveform_family_of_pe_params(f1010_low, f1010_samp, di, eta_low, eta_high, Neta, lamt_low, lamt_high, Nlamt, dlamt_low, dlamt_high, Ndlamt, pn_type='TaylorT4'):
    """
        Generates list of waveform parameters and the corresponding waveforms.
        
        Parameters
        ----------
        di : int
        Downsampling factor. 1 does no downsampling, 2 uses every other point, 2**6 uses one in 64 points, etc.
        
        Returns
        -------
        param_list : 2d list
        waveform_list : list
        List of waveforms
        """
    
    waveform_list = []
    param_list = []
    for eta in np.linspace(eta_low, eta_high, Neta):
        for lamt in np.linspace(lamt_low, lamt_high, Nlamt):
            for dlamt in np.linspace(dlamt_low, dlamt_high, Ndlamt):
                # Generate the waveform
                waveform = pn_waveform_of_pe_params(f1010_low, f1010_samp, eta, lamt, dlamt, pn_type)
                # Downsample the waveform
                #waveform_down = downsample_time_series_from_end(waveform, di)
                waveform_down = downsample_time_series_include_max_amp(waveform, di)
                waveform_list.append(waveform_down)
                param_list.append([eta, lamt, dlamt])
    
    #return param_list, waveform_list
    
    return tsutils.TimeSeriesSet(waveforms=waveform_list, parameters=param_list, regular_grid_shape=[Neta, Nlamt, Ndlamt])
