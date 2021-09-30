"""
duration.py

This script measures the duration of a seismogram using the coda envelope

m.m. holt – september 2021
"""

# Installed imports
import obspy
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from threadpoolctl import threadpool_limits
from scipy.optimize import lsq_linear as lsq

# Local imports
import madpy.config as config
import madpy.plotting as plot
import madpy.processing as proc

import matplotlib.pyplot as plt


def measure_duration(st, ptype='log', save_output=False, outfile=None):
    
    """
    each trace needs event and station location info
    """
    
    # Initialize table
    output = []
    
    for tr in st:
    
        # test that all necessary information exists in trace
        # check that information is correct datatype
            # can use assert statements for these
        # check that outfile specified if save_output is True
        
        # Measure noise
        noise = proc.rms_noise(tr.copy(), 'duration')
        
        # Measure duration
        duration = coda_duration(tr.copy(), noise, ptype)
        
        # Add to data #np.sqrt((amplitude/snr)**2)
        output.append([str(tr.stats.o.date), str(tr.stats.o.time)[:11],
                       tr.stats.elat, tr.stats.elon, tr.stats.edep,
                       tr.stats.station, tr.stats.channel,
                       tr.stats.slat, tr.stats.slon, tr.stats.selev,
                       duration[0], duration[1]])
        
    # Format output
    df = format_output(output)
    
    # Save if desired
    if save_output:
        df.to_csv(f'{outfile}', float_format='%0.5f', index=False)
        
    return df
            

def coda_duration(tr, noise, ptype):
    
    """
    trim waveform and measure coda envelope decay
    """

    dcfg = config.Duration()
    avg_time, avg_data_lin = moving_average(tr, dcfg)
    avg_data_log = log_envelope(avg_data_lin)
    fit_begin, fit_end = coda_fitting_window(tr, dcfg, avg_time, avg_data_log, noise)
    coda_line = coda_line_fit(avg_time[fit_begin:fit_end], avg_data_log[fit_begin:fit_end])
    dur, i_dur = get_duration_index(dcfg, coda_line, noise)
    cc = get_correlation_coefficient(coda_line, avg_time, avg_data_log, fit_begin, i_dur)
    
    # plot
    cfg = config.Duration()
    if cfg.dur_plot == 'yes':
        plot.duration_plot(tr, avg_time, avg_data_lin, avg_data_log, fit_begin, fit_end, dur, cc, coda_line, noise, ptype)
#     (tr, avg_time, avg_data_lin, avg_data_log, fit_start, fit_end, dur, cc, coda, noise, ptype):
   
    return dur, cc


def moving_average(tr, cfg):
    
    """
    do moving average of data
    """
    
    mov_avg_wind_samp = cfg.moving_average_window * int(tr.stats.sampling_rate)
    time_wrt_p = time_relative_p(tr)
    time_averaging = pd.Series(time_wrt_p).rolling(window=mov_avg_wind_samp).median()
    averaged_time = time_averaging.iloc[mov_avg_wind_samp-1:].values
    data_averaging = pd.Series(np.abs(hilbert(tr.data))).rolling(window=mov_avg_wind_samp).mean()
    averaged_data = data_averaging.iloc[mov_avg_wind_samp-1:].values
    
    return averaged_time, averaged_data


def time_relative_p(tr):
    
    """
    Get time with respect to P- arrival
    """
    
    p_wrt_b = (tr.stats.o - tr.stats.starttime) + tr.stats.p
    
    return np.arange(0, len(tr.data)) * tr.stats.delta - p_wrt_b


def log_envelope(data):
    
    """
    transforms coda into envelope
    """
    
    log_env = np.log10(data)
    log_env[np.isinf(log_env)] = np.nan
    log_env[np.isneginf(log_env)] = np.nan
    
    return log_env


def coda_fitting_window(tr, cfg, time, data, noise):
    
    
    """
    find where to start and stop the coda by index
    """
    
    search_window_begin = search_window(tr, time, cfg, 'begin')
    search_window_end = search_window(tr, time, cfg, 'end')
    i_max = np.where(data == np.nanmax(data[search_window_begin:search_window_end]))[0][0]
    i_end = fitting_window_end(cfg, data, i_max, noise)
    i_begin = fitting_window_start(cfg, i_max, i_end)
    
    return i_begin, i_end


def search_window(tr, time, cfg, position):
    
    """
    get indices to search for coda fit values
    """
    
    search_seconds = search_window_seconds(tr, cfg, position)
    search_index = search_window_index(time, search_seconds)
    
    return search_index


def search_window_seconds(tr, cfg, position):
    
    """
    fitting window in seconds relative to p
    """
    
    phase_p = phase_relative_p(tr, cfg.signal_phase)
    
    if position == 'begin':
        seconds_wrt_p = phase_p + cfg.signal_window_begin
    elif position == 'end':
        seconds_wrt_p = phase_p + cfg.signal_window_end
    
    return seconds_wrt_p

        
def phase_relative_p(tr, phase):
    
    """
    get the seconds relative to p
    """
    
    # assert phase
    if phase == 'O':
        phase_time = tr.stats.o - tr.stats.p
    elif phase == 'P':
        phase_time = tr.stats.p - tr.stats.p
    elif phase == 'S':
        phase_time = tr.stats.s - tr.stats.p
        
    return phase_time


def search_window_index(time, time_sec):
    
    """
    fitting window as an index 
    """
    
    return np.where(np.sign(time - time_sec) == 1)[0][0]


def fitting_window_start(cfg, i_max, i_end):
    
    """
    find where it starts
    """
    
    if cfg.start_fit_wrt_max > 1:
        i_begin = (i_end - i_max)/cfg.start_fit_wrt_max + i_max
    else:
        i_begin = np.copy(i_max)
        
    return int(i_begin)


def fitting_window_end(cfg, data, i_max, noise):
                                   
    """
    find where it reaches the desired threshold wrt pre-p noise level
    """
    
    noise_diff = data[i_max:] - np.log10(cfg.end_fit_wrt_noise * noise)
    i_noise = np.where(np.sign(noise_diff) == -1)[0]
    # check i_noise
    
    return int(i_noise[0] + i_max)


def coda_line_fit(x, d):
    
    """
    performs the line fit (with limits for coda)
    """
    
    ones = np.ones(np.shape(x))
    logs = np.log10(x)
    G = np.stack((ones, x), axis=-1)
    
    with threadpool_limits(limits=1, user_api='blas'):
        m = lsq(G, d, bounds=([-np.inf, -np.inf], [np.inf, 0]))
        
    return m.x


def get_duration_index(cfg, line, noise):
    
    """
    find intersection with predetermined level
    """
    
    x_extended, y_extended = extend_line(line[1], line[0], 0, 10000, 0.001) 
    threshold = coda_line_end(cfg, noise)
    cross = np.where(np.sign(y_extended - threshold) == -1)[0]
    # check cross
    
    return x_extended[cross[0]], cross[0]


def extend_line(m, b, xmin, xmax, xint):
    
    x = np.arange(xmin, xmax, xint)
    y = m * x + b
    
    return x, y                   
                    
    
def coda_line_end(cfg, noise):
    
    """
    determine what the duration is
    """
    
    if cfg.end_fit_threshold == 'absolute':
        threshold = cfg.duration_absolute_threshold
    elif cfg.end_fit_threshold == 'noise':
        threshold = cfg.duration_prep_noise * noise
        
    return threshold
 

def get_correlation_coefficient(line, time, data, data_begin, line_end):
    
    """
    get correlation coefficient
    """
    
    x_extended, y_extended = extend_line(line[1], line[0], 0, 10000, 0.001)
    line_begin = np.where(np.sign(x_extended - time[data_begin]) == 1)[0][0]
    data_end = np.where(np.sign(time - x_extended[line_end]) == 1)[0][0]
    time_rounded = np.round(time, 3)
    _, i_line, i_data = np.intersect1d(x_extended[line_begin:line_end], time_rounded[data_begin:data_end], return_indices=True)
    x_data = time[i_data + data_begin]
    y_data = data[i_data + data_begin]
    x_line = x_extended[i_line + line_begin]
    y_line = y_extended[i_line + line_begin]
    true_values = np.column_stack((x_data, x_line))
    pred_values = np.column_stack((10**y_data, 10**y_line))
    cc = calculate_cc(true_values, pred_values)[0, 3]
    
    return cc
    
    
def calculate_cc(y_true, y_pred):
    
    """
    calculates pearson(?) cc
    """
    
    cc = np.corrcoef(y_true, y_pred, rowvar=False)
    # check
    
    return cc   
       
    
def format_output(data):
    
    """
    turn list into dataframe
    """
    
    column_names = ['date', 'time', 'event_lat', 'event_lon', 'event_depth',
                    'station', 'channel', 'station_lat', 'station_lon', 'station_elev',
                    'duration', 'cc']
    
    df = pd.DataFrame(data, columns=column_names)
    
    return df
    