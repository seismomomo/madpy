"""
dur.py

plot duration measurement
"""

import numpy as np
from matplotlib import markers
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.path import Path
import madpy.duration as duration
import madpy.plotting.utils as util
import madpy.plotting.params as params


PLOT_PHASE = 'O'

def duration_plot(tr, avg_time, avg_data_lin, avg_data_log, 
                  fit_start, fit_end, dur, cc, 
                  coda, noise, ptype, cfg):
    """Generate duration plot"""
    
    # plot info
    time = util.to_time(tr, PLOT_PHASE)
    idx_phase = util.relative_phase_indices(tr, time, PLOT_PHASE)
    idx_coda = relative_coda_indices(time, avg_time, 
                                     fit_start, fit_end, idx_phase, dur)
    coda_line = coda_fit_line(time, coda, idx_coda, idx_phase[1])
    noise_threshold = duration.coda_line_end(cfg, noise)
    xinfo = util.format_xaxis(time, -1, 5, 5, 
                              idx_phase[0], idx_coda[2], '{:0.0f}')
    
    # plot parameters
    pp = params.duration_plot_parameters()
    xlabel = 'Time'
    title = r'{:s}: $\tau$ = {:0.3f} s, CC = –{:0.2f}'.format(tr.id, dur, np.abs(cc))
    if ptype == 'linear':
        full_data = tr.data
        avg_data = avg_data_lin
        yinfo = format_duration_yaxis(time, tr, xinfo, 20, 2, 
                                      ptype, '{:0.1E}')
        ylabel = 'Ground motion (linear)'
    elif ptype == 'log':
        full_data = duration.log_envelope(tr.data)
        avg_data = avg_data_log
        yinfo = format_duration_yaxis(time, tr, xinfo, 20, 4, 
                                      ptype, '{:0.0f}')
        ylabel = 'Ground motion (log)'
        
    # plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, full_data, c=pp['c']['dat'], lw=pp['lw']['dat'], zorder=1)
    if cfg.moving_average_window > 0:
        ax.plot(avg_time + time[idx_phase[1]], avg_data, c=pp['c']['avg'], 
                lw=pp['lw']['avg'], zorder=2)
    ax.vlines(time[idx_phase[1]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=3)
    ax.vlines(time[idx_phase[2]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['pha'], color=pp['c']['pha'], zorder=4)
    ax.vlines(time[idx_coda[2]], yinfo['min'], yinfo['max'], 
              lw=pp['lw']['dur'], color=pp['c']['dur'], zorder=5)
    if ptype == 'log':
        ax.hlines(noise_threshold, xinfo['min'], xinfo['max'], 
                  lw=pp['lw']['nth'], color=pp['c']['nth'], zorder=6)
        ax.plot(time, coda_line[0], 
                lw=pp['lw']['fit'], color=pp['c']['fit'], zorder=7)
        if coda_line[1] is not None:
            ax.plot(time, coda_line[1], lw=pp['lw']['fit'], 
                    color=pp['c']['fit'], ls='--', zorder=8)
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
        fig.savefig(f'{cfg.figure_path}/dur-{ptype}-{tr.id}.png')
    
               
def format_duration_yaxis(time, tr, xinfo, yspace, nint, ptype, label_format):
    """Consolidate duration y-axis information"""

    i_xmin = np.where(np.abs(time - xinfo['min']) == \
                      np.nanmin(np.abs(time - xinfo['min'])))[0][0]
    i_xmax = np.where(np.abs(time - xinfo['max']) == \
                      np.nanmin(np.abs(time - xinfo['max'])))[0][0]
    if ptype == 'linear':
        ybig = np.nanmax(np.abs(tr.data[i_xmin:i_xmax]))
        ymin = -ybig - ybig / yspace
        ymax_0 = ybig + ybig / yspace
        yint = ymax_0 / nint
        yticks = util.tick_info(ymin, ymax_0, yint, label_format)
        ymax = np.max(yticks[0])
        yinfo = {'min': ymin, 'max': ymax, 'int': yint,
                 'ticks': yticks[0], 'ticklabels': yticks[1]}
    elif ptype == 'log':
        data = duration.log_envelope(np.abs(hilbert(tr.data)))
        ymin = int(np.ceil(np.nanmin(data[i_xmin:i_xmax])))
        ymax = int(np.ceil(np.nanmax(data[i_xmin:i_xmax])))
        yint = (ymax - ymin) / nint
        yticks = util.tick_info(ymin, ymax, yint, label_format)
        yinfo = {'min': ymin, 'max': ymax, 'int': yint,
                 'ticks': yticks[0], 'ticklabels': yticks[1]}
    
    return yinfo    
    
    
def coda_fit_line(time, coda, idx_coda, i_p, extrapolation=None):
    """Get best fit line and whether or not it's extrapolated"""
    
    y_data = coda[1] * (time - time[i_p]) + coda[0]
    fit_line = np.empty(len(time),) * np.nan
    fit_line[idx_coda[0]:idx_coda[1]] = y_data[idx_coda[0]:idx_coda[1]]

    if idx_coda[2] > idx_coda[1]:
        extrapolation = np.empty(len(time),) * np.nan
        extrapolation[idx_coda[1]:idx_coda[2]] = \
        y_data[idx_coda[1]:idx_coda[2]]        
    
    return fit_line, extrapolation
        
        
def relative_coda_indices(time, avg_time, fit_start, fit_end, idx_phase, dur): 
    """Get indices of coda fit"""
    
    p_relative = time[idx_phase[1]]
    avg_time_rel = avg_time + p_relative
    start_rel = avg_time_rel[fit_start]
    end_rel = avg_time_rel[fit_end]
    dur_rel = dur + p_relative
    i_dur = np.where(np.abs(time - dur_rel) == \
                     np.nanmin(np.abs(time - dur_rel)))[0][0]
    i_start = np.where(np.abs(time - start_rel) == \
                       np.nanmin(np.abs(time - start_rel)))[0][0]
    i_end = np.where(np.abs(time - end_rel) == \
                     np.nanmin(np.abs(time - end_rel)))[0][0]
    # TO-DO: Check indices
    
    return i_start, i_end, i_dur