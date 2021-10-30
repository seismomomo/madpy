"""
amp.py

plot amplitude measurement
"""

import numpy as np
from matplotlib import markers
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.path import Path
import madpy.plotting.utils as util
import madpy.plotting.params as params


PLOT_PHASE = 'O'

def amplitude_plot(tr_full, tr_signal, amp, indices, noise, cfg):
    """Generate amplitude plot"""
    
    # plot info
    time = util.to_time(tr_full, PLOT_PHASE)
    idx_phase = util.relative_phase_indices(tr_full, time, PLOT_PHASE)
    idx_p2p = relative_p2p_indices(tr_full, tr_signal, indices)
    xinfo = util.format_xaxis(time, -2, 5, 5, idx_phase[1], idx_p2p[1], '{:0.0f}')
    yinfo = format_amplitude_yaxis(time, tr_full, xinfo, 20, 2, '{:0.1E}')

    # plot parameters
    pp = params.amplitude_plot_parameters()
    xlabel = 'Time'
    ylabel = 'Ground motion'
    title = '{}: A = {:0.3f}, SNR = {:0.3f}'.format(
        tr_full.id, amp, amp / noise)    
    
    # plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hlines(tr_full.data[idx_p2p[0]], xinfo['min'], xinfo['max'], 
              lw=pp['lw']['p2pl'], color=pp['c']['p2pl'], zorder=1)
    ax.hlines(tr_full.data[idx_p2p[1]], xinfo['min'], xinfo['max'], 
              lw=pp['lw']['p2pl'], color=pp['c']['p2pl'], zorder=2)
    ax.plot(time, tr_full.data, c=pp['c']['dat'], 
            lw=pp['lw']['dat'], zorder=3)
    ax.vlines(time[idx_phase[0]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=4)
    ax.vlines(time[idx_phase[1]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=5)
    ax.vlines(time[idx_phase[2]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=6)
    ax.plot(time[idx_p2p[0]], tr_full.data[idx_p2p[0]],
            marker=util.align_marker('^', valign='top'), c=pp['c']['p2pm'], 
            ms=18, alpha=0.5, zorder=7)
    ax.plot(time[idx_p2p[1]], tr_full.data[idx_p2p[1]],
            marker=util.align_marker('v', valign='bottom'), c=pp['c']['p2pm'], 
            ms=18, alpha=0.5, zorder=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xinfo['min'], xinfo['max'])
    ax.set_xticks(xinfo['ticks'])
    ax.set_xticklabels(xinfo['ticklabels'])
    ax.set_ylabel(ylabel)
    ax.set_ylim(yinfo['min'], yinfo['max'])
    ax.set_yticks(yinfo['ticks'])
    ax.set_yticklabels(yinfo['ticklabels'])
    ax.set_title(title)
    plt.tight_layout()
    plt.close()
    
    if cfg.save_figure:
        fig.savefig(f'{cfg.figure_path}/amp-{tr_full.id}.png')
    
    
def relative_p2p_indices(tr_full, tr_signal, indices):
    """Get peak-to-peak indices relative to PLOT_PHASE"""
    
    i_signal = np.where(np.in1d(tr_full.data, tr_signal.data))[0]
    i_peak1 = i_signal[0] + indices[0]
    i_peak2 = i_signal[0] + indices[1]
    # TO-DO: check indices 
    indices_ordered = order_indices(tr_full, i_peak1, i_peak2) 
    
    return indices_ordered


def order_indices(tr, i_peak1, i_peak2):
    """Order indices such that the lowest one is first"""
    
    peak1 = tr.data[i_peak1]
    peak2 = tr.data[i_peak2]
    if peak1 > peak2:
        indices_ordered = np.array([i_peak2, i_peak1])
    else:
        indices_ordered = np.array([i_peak1, i_peak2])
        
    return indices_ordered


def format_amplitude_yaxis(time, tr, xinfo, yspace, nint, label_format):
    """Consolidate amplitude y-axis information"""
    
    i_xmin = np.where(np.abs(time - xinfo['min']) == \
                      np.nanmin(np.abs(time - xinfo['min'])))[0][0]
    i_xmax = np.where(np.abs(time - xinfo['max']) == \
                      np.nanmin(np.abs(time - xinfo['max'])))[0][0]
    ybig = np.max(np.abs(tr.data[i_xmin:i_xmax]))
    ymin = -ybig - ybig / yspace
    ymax_0 = ybig + ybig / yspace
    yint = ymax_0 / nint
    yticks = util.tick_info(ymin, ymax_0, yint, label_format)
    ymax = np.max(yticks[0])
    yinfo = {'min': ymin, 'max': ymax, 'int': yint,
             'ticks': yticks[0], 'ticklabels': yticks[1]}
    
    return yinfo