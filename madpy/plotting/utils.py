"""
utils.py

utility functions for plotting
"""

import numpy as np
import madpy.noise as n
from matplotlib import markers
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.path import Path


def to_time(tr, phase):
    """create a time axis"""
    
    arrival = n.arrival_time_utc(tr, phase)
    shift = arrival - tr.stats.starttime
    samples = np.arange(0, len(tr)) * tr.stats.delta
    
    return samples - shift


def relative_phase_indices(tr, time, phase):
    """return indices for o, p, and s"""
    
    p_utc = n.arrival_time_utc(tr, 'P')
    s_utc = n.arrival_time_utc(tr, 'S')
    ref = n.arrival_time_utc(tr, phase)
    o_rel = tr.stats.o - ref
    p_rel = p_utc - ref
    s_rel = s_utc - ref
    i_o = np.where(np.abs(time - o_rel) == \
                   np.nanmin(np.abs(time - o_rel)))[0][0]
    i_p = np.where(np.abs(time - p_rel) == \
                   np.nanmin(np.abs(time - p_rel)))[0][0]
    i_s = np.where(np.abs(time - s_rel) == \
                   np.nanmin(np.abs(time - s_rel)))[0][0]
    # TO–DO: check indices
    
    return i_o, i_p, i_s


def format_xaxis(time, t1, t2, nint, ref1, ref2, label_format):
    """get xlimits and ticks relative to ref1 and ref2"""
    
    xmin = int(np.ceil(time[ref1] + t1))
    xmax_0 = int(np.floor(time[ref2] + t2))
    xint = int(np.round((xmax_0 - xmin) / 5))
    xticks = tick_info(xmin, xmax_0, xint, label_format)
    xmax = np.max(xticks[0])
    
    xinfo = {'min': xmin, 'max': xmax, 'int': xint,
             'ticks': xticks[0], 'ticklabels': xticks[1]}
    
    return xinfo
    
    
def tick_info(axmin, axmax, axint, label_format):
    """get tick info including labels"""
    
    axticks = np.arange(axmin, axmax + axint, axint)
    axticklabels = [label_format.format(tick) for tick in axticks]
    
    return axticks, axticklabels

    
def align_marker(marker, halign='center', valign='middle'):
    """Align matplotlib markers
    
    Code taken from 
    https://stackoverflow.com/questions/26686722/align-matplotlib-scatter-marker-left-and-or-right
    
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

    bm = markers.MarkerStyle(marker)
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)