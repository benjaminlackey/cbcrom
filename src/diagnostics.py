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

import timeseriesutils as tsutils

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_parameters(axes, xindex, yindex, ts_params=None, rb_params=None, xlabel=None, ylabel=None):
    """Plot the training-set parameters and/or the reduced-basis parameters.
    
    Parameters
    ----------
    axes : figure axes
    xindex : int
        Index for the x-axis data
    yindex : int
        Index for the y-axis data
    ts_params : 2d arraylike
        List of training set parameters.
    rb_params : 2d arraylike
        List of reduced basis parameters.
    xlabel : string
        Label for the x-axis.
    ylabel : string
        Label for the y-axis.
    """
    
    if xlabel is None:
        xlabel = 'Parameter '+str(xindex)
    if ylabel is None:
        ylabel = 'Parameter '+str(yindex)
    
    # Plot the parameters of the training set
    if ts_params:
        ts = np.array(ts_params)
        axes.scatter(ts[:, xindex], ts[:, yindex], marker='x', s=10, c='r')
    
    # Plot the parameters used to generate reduced basis waveforms and number them
    if rb_params:
        rb = np.array(rb_params)
        for i in range(len(rb)):
            axes.scatter(rb[i, xindex], rb[i, yindex], marker='$'+str(i)+'$', s=100, color='b')
    
    #axes.set_xlim([0.218, 0.252])
    #axes.set_ylim([-200, 1.1*lamt_high])
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)


def check_orthonormality(waveforms, inner_product, plot=True):
    """Check whether or not a list of waveforms is actually orthonormal.
    Then make a plot.
    
    Parameters
    ----------
    waveforms : List of waveform objects
        The objects can be TimeSeries, FrequencySeries, etc.
    inner_product : function(h1, h2)
        Function that evaluates the inner product (scalar) of two waveforms h1 and h2.
        
    Returns
    -------
    inner_ij : 2d-array
        absolute values of the inner products |<h_i|h_j>|
    """
    
    Nbases = len(waveforms)
    
    inner_ij = np.array([[np.abs(inner_product(waveforms[i], waveforms[j])) for i in range(Nbases)] for j in range(Nbases)])
    
    if plot:
        # Plot matrix of inner products
        fig = plt.figure(figsize=(8, 8))
        axes = fig.add_subplot(111)
        im = axes.imshow(inner_ij, cmap=plt.get_cmap('hot'), interpolation='nearest', norm=LogNorm(vmin=1.0e-16, vmax=1.0))
        cb = plt.colorbar(mappable=im, ax=axes)
        cb.set_label(label=r'$|\langle h_i, h_j \rangle|$', fontsize=16)
        cb.ax.tick_params(labelsize=14)
        axes.set_xlabel(r'$i$', fontsize=16)
        axes.set_ylabel(r'$j$', fontsize=16)
    
    return inner_ij


def plot_greedy_error(axes, sigma_list):
    """Plot the greedy error as a function of the number of reduced bases.
    
    Parameters
    ----------
    axes : axes object
        The axes you want to put the plot on.
    sigma_list : list of floats
        The greedy errors
    """
    
    axes.semilogy(sigma_list, color='k', ls='-', lw=1.5)
    #axes.semilogy(sigma_list, color='b', marker='o')
    
    axes.set_xlabel(r'Waveform index $m$', fontsize=16)
    axes.set_ylabel(r'Greedy error $\sigma_m$', fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)


################################################################################
#             Testing quality of amplitude and phase functions
################################################################################

def plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=None, xlabel=None, zlabel=None, ylabel=None, logy=None):
    """
        Plot a cross section of the data on a grid.
        
        Parameters
        ----------
        axes : matplotlib axes
        param_grid : N+1 dimension numpy array
        Parameter values at the grid points
        data_grid : N dimension numpy array
        Data at the grid points
        slices : tuple of length Ndim
        Use integers to specify the indices for the parameters that are fixed.
        Use 'x' for the parameter that will be varied on the x axis.
        Use 'z' for the parameter that will be varied on different curves.
        Ex: (7, 'z', 'x', 5) will hold the 0th and 3rd indices fixed,
        vary the 2nd index on the x axis, and vary the 1st index on different curves.
        function : func(params)
        Optional fitting/interpolating function to plot on top of the data.
        xlabel : string
        Name for the x axis.
        ylabel : string
        Name for the y axis.
        zlabel : string
        Name for the parameter that you vary on different curves (z axis).
        """
    
    # Get the axis corresponding to the parameter you want on the x-axis
    xaxis = slices.index('x')
    # Get the axis corresponding to the parameter you want to vary on different curves
    zaxis = slices.index('z')
    Nx = data_grid.shape[xaxis]
    Nz = data_grid.shape[zaxis]
    
    #print xaxis, zaxis, data_grid.shape, Nx, Nz
    
    for iz in range(Nz):
        # Construct the indices for the x-axis slice
        xslice = list(slices)
        xslice[xaxis] = slice(Nx)
        xslice[zaxis] = iz
        parameters = param_grid[tuple(xslice)]
        xparameters = parameters[:, xaxis]
        data = data_grid[tuple(xslice)]
        # Plot the data
        if logy:
            axes.plot(xparameters, np.log10(data), 'o', ms=5, ls=':')
        else:
            axes.plot(xparameters, data, 'o', ms=5, ls=':')
        # Plot the approximant function if given
        if function:
            parameters = np.array([parameters[0] for i in range(100)])
            xparameters = np.linspace(xparameters[0], xparameters[-1], 100)
            parameters[:, xaxis] = xparameters
            farray = map(function, parameters)
            if logy:
                axes.plot(xparameters, np.log10(farray), c='k', ls='-', lw=1)
            else:
                axes.plot(xparameters, farray, c='k', ls='-', lw=1)

    axes.set_title(xlabel+' varies on x-axis.\n'+zlabel+' varies on different curves.\nOther parameter(s) are fixed.', fontsize=16)
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)


def plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type='diff', xlabel=None, zlabel=None, ylabel=None, logy=None):
    """
        Plot the error in a function that approximates the data on the grid.
        
        Parameters
        ----------
        axes : matplotlib axes
        param_grid : N+1 dimension numpy array
        Parameter values at the grid points
        data_grid : N dimension numpy array
        Data at the grid points
        slices : tuple of length Ndim
        Use integers to specify the indices for the parameters that are fixed.
        Use 'x' for the parameter that will be varied on the x axis.
        Use 'z' for the parameter that will be varied on different curves.
        Ex: (7, 'z', 'x', 5) will hold the 0th and 3rd indices fixed,
        vary the 2nd index on the x axis, and vary the 1st index on different curves.
        function : func(params)
        Optional fitting/interpolating function to plot on top of the data.
        error_type : string
        'diff' for difference. 'frac' for fractional error.
        xlabel : string
        Name for the x axis.
        ylabel : string
        Name for the y axis.
        zlabel : string
        Name for the parameter that you vary on different curves (z axis).
        """
    
    # Evaluate the function on the parameter grid
    grid_shape = param_grid.shape[:-1]
    Nparams = param_grid.shape[-1]
    Npoints = np.prod(grid_shape)
    param_array = param_grid.reshape(Npoints, Nparams)
    func_array = np.array(map(function, param_array))
    func_grid = func_array.reshape(grid_shape)
    
    # Evaluate the error
    if error_type == 'diff':
        error_grid = func_grid - data_grid
    elif error_type == 'frac':
        error_grid = func_grid/data_grid - 1.0
    else:
        raise Exception, """error_type must be either 'diff' or 'frac'."""
    
    plot_cross_sections_of_grid(axes, param_grid, error_grid, slices, function=None, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)


def compare_grid_to_approximant_on_cross_sections_eta_lam1_lam2(param_grid, amp_grid, phase_grid, amp_func, phase_func, ifixeta, ifixlam1, ifixlam2, logy=None):
    
    ################### Plot amplitude ##################
    data_grid = amp_grid
    function = amp_func
    ylabel = r'$A(T_j)$'
    
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', ifixlam2)
    xlabel, zlabel = r'$\eta$', r'$\Lambda_1$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', ifixlam2)
    xlabel, zlabel = r'$\Lambda_1$', r'$\eta$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(143)
    slices = ('z', ifixlam1, 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\eta$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(144)
    slices = (ifixeta, 'z', 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\Lambda_1$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    ################## Plot fractional amplitude error ##################
    error_type = 'frac'
    ylabel = r'$A_{\rm approx}/A - 1$'
    
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', ifixlam2)
    xlabel, zlabel = r'$\eta$', r'$\Lambda_1$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', ifixlam2)
    xlabel, zlabel = r'$\Lambda_1$', r'$\eta$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(143)
    slices = ('z', ifixlam1, 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\eta$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(144)
    slices = (ifixeta, 'z', 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\Lambda_1$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    ################## Plot phase ####################
    data_grid = phase_grid
    function = phase_func
    ylabel = r'$\Phi(T_j)$'
    
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', ifixlam2)
    xlabel, zlabel = r'$\eta$', r'$\Lambda_1$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', ifixlam2)
    xlabel, zlabel = r'$\Lambda_1$', r'$\eta$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(143)
    slices = ('z', ifixlam1, 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\eta$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    axes = fig.add_subplot(144)
    slices = (ifixeta, 'z', 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\Lambda_1$'
    plot_cross_sections_of_grid(axes, param_grid, data_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel, logy=logy)
    
    ################## Plot phase error ##################
    error_type = 'diff'
    ylabel = r'$\Phi_{\rm approx} - \Phi$'
    
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', ifixlam2)
    xlabel, zlabel = r'$\eta$', r'$\Lambda_1$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', ifixlam2)
    xlabel, zlabel = r'$\Lambda_1$', r'$\eta$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(143)
    slices = ('z', ifixlam1, 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\eta$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(144)
    slices = (ifixeta, 'z', 'x')
    xlabel, zlabel = r'$\Lambda_2$', r'$\Lambda_1$'
    plot_remainder_for_cross_sections_of_grid(axes, param_grid, data_grid, slices, function, error_type=error_type, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)



def compare_grid_to_approximant_on_cross_sections_eta_lamt_dlamt(pe_param_grid, pe_amp_grid, pe_phase_grid, pe_amp_func, pe_phase_func, i0, i1, i2):
    ################### Plot amplitude ##################
    function = pe_amp_func
    ylabel = r'$A(T_j)$'
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', i2)
    xlabel, zlabel = r'$\eta$', r'$\tilde\Lambda$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_amp_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', i2)
    xlabel, zlabel = r'$\tilde\Lambda$', r'$\eta$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_amp_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(143)
    slices = ('z', i1, 'x')
    xlabel, zlabel = r'$\delta\tilde\Lambda$', r'$\eta$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_amp_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(144)
    slices = (i0, 'z', 'x')
    xlabel, zlabel = r'$\delta\tilde\Lambda$', r'$\tilde\Lambda$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_amp_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    
    ################### Plot phase ##################
    function = pe_phase_func
    ylabel = r'$\Phi(T_j)$'
    fig = plt.figure(figsize=(24, 6))
    
    axes = fig.add_subplot(141)
    slices = ('x', 'z', i2)
    xlabel, zlabel = r'$\eta$', r'$\tilde\Lambda$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_phase_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(142)
    slices = ('z', 'x', i2)
    xlabel, zlabel = r'$\tilde\Lambda$', r'$\eta$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_phase_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(143)
    slices = ('z', i1, 'x')
    xlabel, zlabel = r'$\delta\tilde\Lambda$', r'$\eta$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_phase_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)
    
    axes = fig.add_subplot(144)
    slices = (i0, 'z', 'x')
    xlabel, zlabel = r'$\delta\tilde\Lambda$', r'$\tilde\Lambda$'
    plot_cross_sections_of_grid(axes, pe_param_grid, pe_phase_grid, slices, function=function, xlabel=xlabel, zlabel=zlabel, ylabel=ylabel)



################################################################################
#                Find maximum error of a ROM                                   #
################################################################################

def max_amp_phase_error(htrue, hrom):
    
    # Get amplitude and phase of each waveform
    amp_true, phase_true = tsutils.amp_phase_from_complex(htrue)
    amp_rom, phase_rom = tsutils.amp_phase_from_complex(hrom)
    
    # Find index of max amplitude
    true_max, true_maxi = htrue.abs_max_loc()
    rom_max, rom_maxi = hrom.abs_max_loc()
    
    # Calculate |relative-amplitude error| and |phase-difference error|
    # Dropping the points after max amplitude that have zero amplitude
    amperror = np.abs(amp_rom.numpy()[:true_maxi]/amp_true.numpy()[:true_maxi]-1)
    phaseerror = np.abs(phase_rom.numpy()[:true_maxi] - phase_true.numpy()[:true_maxi])
    
    iamperrmax = np.abs(amperror).argmax()
    tamperrmax = htrue.sample_times[iamperrmax]
    amperrmax = amperror[iamperrmax]
    
    iphaseerrmax = np.abs(phaseerror).argmax()
    tphaseerrmax = htrue.sample_times[iphaseerrmax]
    phaseerrmax = phaseerror[iphaseerrmax]
    
    return iamperrmax, tamperrmax, amperrmax, iphaseerrmax, tphaseerrmax, phaseerrmax



######### Plot projection of the maximum error ##########

def max_error_2d_projection_plot(axes, params_list, error_list, xi, yi, x_label, y_label, rb_params_list=None, colorbar=None, colorbarlabel='cblabel'):
    
    # Sort errors so largest errors are plotted on top of smaller errors
    error_params = np.hstack((np.array([error_list]).T, np.array(params_list)))
    error_params_sort = error_params[error_params[:, 0].argsort()]
    error_max = error_params_sort[-1, 0]
    
    # Scatter plot with colorbar
    sc = axes.scatter(error_params_sort[:, xi+1], error_params_sort[:, yi+1], c=error_params_sort[:, 0], s=100*error_params_sort[:, 0]/error_max,
                      edgecolor='', alpha=1.0)
        
    if colorbar:
        cb = plt.colorbar(mappable=sc, ax=axes)
        cb.set_label(label=colorbarlabel, fontsize=18)
        cb.ax.tick_params(labelsize=14)

    # Plot parameter values of reduced basis if given
    if rb_params_list is not None:
        rb_params_array = np.array(rb_params_list)
        for i in range(len(rb_params_array)):
            axes.scatter(rb_params_array[i, xi], rb_params_array[i, yi], marker='$'+str(i)+'$', s=100, c='r')

    # buffers for plot
    params_array = np.array(params_list)
    xmin, xmax = min(params_array[:, xi]), max(params_array[:, xi])
    ymin, ymax = min(params_array[:, yi]), max(params_array[:, yi])
    bufx = 0.05*(xmax - xmin)
    bufy = 0.05*(ymax - ymin)
    
    axes.set_xlim([xmin-bufx, xmax+bufx])
    axes.set_ylim([ymin-bufy, ymax+bufy])
    axes.set_xlabel(x_label, fontsize=16)
    axes.set_ylabel(y_label, fontsize=16)
    axes.set_xticklabels(axes.get_xticks(), fontsize=14)
    axes.set_yticklabels(axes.get_yticks(), fontsize=14)
    axes.minorticks_on()
    axes.tick_params(which='major', width=2, length=8)
    axes.tick_params(which='minor', width=2, length=4)

