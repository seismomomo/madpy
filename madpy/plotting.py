"""
plotting.py

stuff

m. m. holt – September 2021
"""

# Installed imports
import numpy as np
from matplotlib import markers
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Local imports
import PlotParameters
import madpy.config as config
import madpy.processing as proc
import madpy.duration as duration
import madpy.amplitude as amplitude

PLOT_PHASE = 'P'


def duration_plot(tr, avg_time, avg_data_lin, avg_data_log, fit_start, fit_end, dur, cc, coda, noise, ptype):
    
    """
    does duration plotting
    """
    
    time = to_time(tr)
    idx_phase = relative_phase_indices(tr, time)
    idx_coda = relative_coda_indices(time, avg_time, fit_start, fit_end, idx_phase, dur)
    coda_line = coda_fit_line(time, coda, idx_coda, idx_phase[1])
    noise_threshold = duration_threshold(noise)
    xinfo = format_xaxis(time, -1, 5, 5, idx_phase[0], idx_coda[2], '{:0.0f}')
    
    # Plot parameters
    pp = duration_plot_parameters()
    xlabel = 'Time relative to origin'
    title = r'$\tau$ = {:0.3f} s, CC = –{:0.2f}'.format(dur, np.abs(cc))
    if ptype == 'linear':
        full_data = tr.data
        avg_data = avg_data_lin
        yinfo = format_duration_yaxis(time, tr, xinfo, 20, 2, ptype, '{:0.1E}')
        ylabel = 'Velocity (m/s)'
    elif ptype == 'log':
        full_data = duration.log_envelope(np.abs(hilbert(tr.data)))
        avg_data = avg_data_log
        yinfo = format_duration_yaxis(time, tr, xinfo, 20, 4, ptype, '{:0.0f}')
        ylabel = r'$log_{10}$ velocity (m/s)'
        
    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, full_data, c=pp['c']['dat'], lw=pp['lw']['dat'], zorder=1)
    ax.plot(avg_time + time[idx_phase[1]], avg_data, c=pp['c']['avg'], lw=pp['lw']['avg'], zorder=2)
    ax.vlines(time[idx_phase[1]], yinfo['min'], yinfo['max'], lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=3)
    ax.vlines(time[idx_phase[2]], yinfo['min'], yinfo['max'], lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=4)
    ax.vlines(time[idx_coda[2]], yinfo['min'], yinfo['max'], lw=pp['lw']['dur'], color=pp['c']['dur'], zorder=5)
    if ptype == 'log':
        ax.hlines(noise_threshold, xinfo['min'], xinfo['max'], lw=pp['lw']['nth'], color=pp['c']['nth'], zorder=6)
        ax.plot(time, coda_line[0], lw=pp['lw']['fit'], color=pp['c']['fit'], zorder=7)
        if coda_line[1] is not None:
            ax.plot(time, coda_line[1], lw=pp['lw']['fit'], color=pp['c']['fit'], ls='--', zorder=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xinfo['min'], xinfo['max'])
    ax.set_xticks(xinfo['ticks'])
    ax.set_xticklabels(xinfo['ticklabels'])
    ax.set_ylabel(ylabel)
    ax.set_ylim(yinfo['min'], yinfo['max'])
    ax.set_yticks(yinfo['ticks'])
    ax.set_yticklabels(yinfo['ticklabels'])
    ax.set_title(title)
    
        
        
def duration_plot_parameters():
    
    """
    duration plotting parameters
    """
    
    colors = {
        'dat': 'dimgray',
        'pha': 'cornflowerblue',
        'dur': 'darkblue',
        'avg': 'lavender',
        'nth': 'forestgreen',
        'fit': 'mediumvioletred'
    }
    
    linewidths = {
        'dat': 1,
        'pha': 2,
        'dur': 2,
        'avg': 2,
        'nth': 2,
        'fit': 2
    }
    
    labels = {
        'dat': 'data',
        'o': 'o-arr',
        'p': 'p-arr',
        's': 's-arr',
        'dur': 'dur',
        'avg': 'avg',
        'nth': 'thresh',
        'fit': 'linefit'
    }

    
    return {'c': colors, 'lw': linewidths, 'lab': labels}

    
    
def format_duration_yaxis(time, tr, xinfo, yspace, nint, ptype, label_format):
    
    """
    get yaxis info stuffs
    """

    i_xmin = np.where(np.abs(time - xinfo['min']) == np.nanmin(np.abs(time - xinfo['min'])))[0][0]
    i_xmax = np.where(np.abs(time - xinfo['max']) == np.nanmin(np.abs(time - xinfo['max'])))[0][0]
    if ptype == 'linear':
        ybig = np.nanmax(np.abs(tr.data[i_xmin:i_xmax]))
        ymin = -ybig - ybig / yspace
        ymax_0 = ybig + ybig / yspace
        yint = ymax_0 / nint
        yticks = tick_info(ymin, ymax_0, yint, label_format)
        ymax = np.max(yticks[0])
        yinfo = {'min': ymin, 'max': ymax, 'int': yint,
                 'ticks': yticks[0], 'ticklabels': yticks[1]}
    elif ptype == 'log':
        data = duration.log_envelope(np.abs(hilbert(tr.data)))
        ymin = int(np.ceil(np.nanmin(data[i_xmin:i_xmax])))
        ymax = int(np.ceil(np.nanmax(data[i_xmin:i_xmax])))
        yint = (ymax - ymin) / nint
        yticks = tick_info(ymin, ymax, yint, label_format)
        yinfo = {'min': ymin, 'max': ymax, 'int': yint,
                 'ticks': yticks[0], 'ticklabels': yticks[1]}
    
    return yinfo    

    
    
def duration_threshold(noise):
    
    """
    determine what the duration threshold is
    """
    
    cfg = config.Duration()
    if cfg.end_fit_threshold == 'absolute':
        threshold = cfg.duration_absolute_threshold
    elif cfg.end_fit_threshold == 'prep_noise':
        threshold = np.log10(cfg.duration_prep_noise * noise)
        
    return threshold        
    
    
def coda_fit_line(time, coda, idx_coda, i_p, extrapolation=None):
    
    """
    get coda best line
    """
    
    y_data = coda[1] * (time - time[i_p]) + coda[0]
    fit_line = np.empty(len(time),) * np.nan
    fit_line[idx_coda[0]:idx_coda[1]] = y_data[idx_coda[0]:idx_coda[1]]

    if idx_coda[2] > idx_coda[1]:
        extrapolation = np.empty(len(time),) * np.nan
        extrapolation[idx_coda[1]:idx_coda[2]] = y_data[idx_coda[1]:idx_coda[2]]
        
    
    return fit_line, extrapolation
        
        
def relative_coda_indices(time, avg_time, fit_start, fit_end, idx_phase, dur): 
    
    """
    indices of coda fit
    """
    
    p_relative = time[idx_phase[1]]
    avg_time_rel = avg_time + p_relative
    start_rel = avg_time_rel[fit_start]
    end_rel = avg_time_rel[fit_end]
    dur_rel = dur + p_relative
    i_dur = np.where(np.abs(time - dur_rel) == np.nanmin(np.abs(time - dur_rel)))[0][0]
    i_start = np.where(np.abs(time - start_rel) == np.nanmin(np.abs(time - start_rel)))[0][0]
    i_end = np.where(np.abs(time - end_rel) == np.nanmin(np.abs(time - end_rel)))[0][0]
    # check indices
    
    return i_start, i_end, i_dur


def amplitude_plot(tr_full, tr_signal, amp, indices, noise):
    
    """
    does the amplitude plotting
    """
    
    time = to_time(tr_full)
    idx_phase = relative_phase_indices(tr_full, time)
    idx_p2p = relative_p2p_indices(tr_full, tr_signal, indices)
    xinfo = format_xaxis(time, -2, 5, 5, idx_phase[1], idx_p2p[1], '{:0.0f}')
    yinfo = format_amplitude_yaxis(time, tr_full, xinfo, 20, 2, '{:0.1E}')

    pp = amplitude_plot_parameters()
    xlabel = 'Time relative to origin'
    ylabel = 'Displacement (mm)'
    title = '{}: A = {:0.3f} mm, SNR = {:0.3f}'.format(tr_full.id, amp, amp / noise)    
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hlines(tr_full.data[idx_p2p[0]], xinfo['min'], xinfo['max'], lw=pp['lw']['p2pl'], color=pp['c']['p2pl'], zorder=1)
    ax.hlines(tr_full.data[idx_p2p[1]], xinfo['min'], xinfo['max'], lw=pp['lw']['p2pl'], color=pp['c']['p2pl'], zorder=2)
    ax.plot(time, tr_full.data, c=pp['c']['dat'], lw=pp['lw']['dat'], zorder=3)
    ax.vlines(time[idx_phase[0]], yinfo['min'], yinfo['max'], lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=4)
    ax.vlines(time[idx_phase[1]], yinfo['min'], yinfo['max'], lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=5)
    ax.vlines(time[idx_phase[2]], yinfo['min'], yinfo['max'], lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=6)
    ax.plot(time[idx_p2p[0]], tr_full.data[idx_p2p[0]],
            marker=align_marker('^', valign='top'), c=pp['c']['p2pm'], ms=18, alpha=0.5, zorder=7)
    ax.plot(time[idx_p2p[1]], tr_full.data[idx_p2p[1]],
            marker=align_marker('v', valign='bottom'), c=pp['c']['p2pm'], ms=18, alpha=0.5, zorder=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xinfo['min'], xinfo['max'])
    ax.set_xticks(xinfo['ticks'])
    ax.set_xticklabels(xinfo['ticklabels'])
    ax.set_ylabel(ylabel)
    ax.set_ylim(yinfo['min'], yinfo['max'])
    ax.set_yticks(yinfo['ticks'])
    ax.set_yticklabels(yinfo['ticklabels'])
    ax.set_title(title)
    
    
def to_time(tr):
    
    """
    create a time axis
    """
    
    arrival = proc.arrival_time_utc(tr, PLOT_PHASE)
    shift = arrival - tr.stats.starttime
    samples = np.arange(0, len(tr)) * tr.stats.delta
    
    return samples - shift


def relative_phase_indices(tr, time):
    
    """
    return indices for o, p, and s
    "ref" must be in seconds relative to the time
    """
    
    p_utc = proc.arrival_time_utc(tr, 'P')
    s_utc = proc.arrival_time_utc(tr, 'S')
    ref = proc.arrival_time_utc(tr, PLOT_PHASE)
    o_rel = tr.stats.o - ref
    p_rel = p_utc - ref
    s_rel = s_utc - ref
    i_o = np.where(np.abs(time - o_rel) == np.nanmin(np.abs(time - o_rel)))[0][0]
    i_p = np.where(np.abs(time - p_rel) == np.nanmin(np.abs(time - p_rel)))[0][0]
    i_s = np.where(np.abs(time - s_rel) == np.nanmin(np.abs(time - s_rel)))[0][0]
    # check indices
    
    return i_o, i_p, i_s


def relative_p2p_indices(tr_full, tr_signal, indices):
    
    """
    get p2p indices relative to full trace
    """
    
    i_signal = np.where(np.in1d(tr_full.data, tr_signal.data))[0]
    i_peak1 = i_signal[0] + indices[0]
    i_peak2 = i_signal[0] + indices[1]
    # check indices 
    indices_ordered = order_indices(tr_full, i_peak1, i_peak2) 
    
    return indices_ordered

def order_indices(tr, i_peak1, i_peak2):
    
    """
    this orders the p2p indices such that the first index in the array is the lowest peak
    """
    
    peak1 = tr.data[i_peak1]
    peak2 = tr.data[i_peak2]
    if peak1 > peak2:
        indices_ordered = np.array([i_peak2, i_peak1])
    else:
        indices_ordered = np.array([i_peak1, i_peak2])
        
    return indices_ordered
    

def format_xaxis(time, t1, t2, nint, ref1, ref2, label_format):
    
    """
    get xlimits and ticks relative to ref1 and ref2
    """
    
    xmin = np.int(np.ceil(time[ref1] + t1))
    xmax_0 = np.int(np.floor(time[ref2] + t2))
    xint = np.int(np.round((xmax_0 - xmin) / 5))
    xticks = tick_info(xmin, xmax_0, xint, label_format)
    xmax = np.max(xticks[0])
    
    xinfo = {'min': xmin, 'max': xmax, 'int': xint,
             'ticks': xticks[0], 'ticklabels': xticks[1]}
    
    return xinfo
    
    
def tick_info(axmin, axmax, axint, label_format):
    
    """
    get tick info including labels
    """
    axticks = np.arange(axmin, axmax + axint, axint)
    axticklabels = [label_format.format(tick) for tick in axticks]
    
    return axticks, axticklabels


def format_amplitude_yaxis(time, tr, xinfo, yspace, nint, label_format):
    
    """
    get yaxis info stuffs
    """
    
    i_xmin = np.where(np.abs(time - xinfo['min']) == np.nanmin(np.abs(time - xinfo['min'])))[0][0]
    i_xmax = np.where(np.abs(time - xinfo['max']) == np.nanmin(np.abs(time - xinfo['max'])))[0][0]
    ybig = np.max(np.abs(tr.data[i_xmin:i_xmax]))
    ymin = -ybig - ybig / yspace
    ymax_0 = ybig + ybig / yspace
    yint = ymax_0 / nint
    yticks = tick_info(ymin, ymax_0, yint, label_format)
    ymax = np.max(yticks[0])
    yinfo = {'min': ymin, 'max': ymax, 'int': yint,
             'ticks': yticks[0], 'ticklabels': yticks[1]}
    
    return yinfo
    

def amplitude_plot_parameters():
    
    """
    amplitude plotting parameters
    """
    
    colors = {
        'dat': 'dimgray',
        'pha': 'cornflowerblue',
        'p2pl': 'forestgreen',
        'p2pm': 'mediumvioletred'
    }
    
    linewidths = {
        'dat': 1,
        'pha': 2,
        'p2pl': 2
    }
    
    labels = {
        'dat': 'data',
        'o': 'o-arr',
        'p': 'p-arr',
        's': 's-arr',
        'p2pm': 'p2p'
    }

    
    return {'c': colors, 'lw': linewidths, 'lab': labels}
    

    
def align_marker(marker, halign='center', valign='middle'):
    """
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    if halign.isalpha():
        halign = {'right': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'left': 1.,
                  }[halign]

    if valign.isalpha():
        valign = {'top': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'bottom': 1.,
                  }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)