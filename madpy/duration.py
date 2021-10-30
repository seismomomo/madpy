"""
duration.py

measure the duration using the coda envelope
"""

import obspy
import types
import warnings
import numpy as np
import pandas as pd
import madpy.noise as n
from typing import Tuple
import madpy.checks as ch
import madpy.config as config
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import madpy.plotting.dur as plot
from threadpoolctl import threadpool_limits
from scipy.optimize import lsq_linear as lsq


def measure_duration(
    st: obspy.Stream, 
    cfg: types.ModuleType = config, 
    ptype: str ='log', 
) -> pd.DataFrame:
    """Measure noise level and duration

    Args:
        st: stream containing one or more time series
        cfg: configuration file
        ptype: plot type [linear, log]
        
    Returns:
        df: dataframe of time series duration information

    """    
    
    output = []   
    for tr in st:
        
        preliminary_checks(tr, ptype, cfg)
        noise = n.rms_noise(tr.copy(), 'duration', cfg)
        duration, cc = coda_duration(tr.copy(), noise, ptype, cfg)
        output.append([str(tr.stats.o.date), str(tr.stats.o.time)[:11],
                       tr.stats.network, tr.stats.station, tr.stats.channel,
                       duration, cc, np.log10(noise)])

    df = format_output(output)
    if cfg.Duration.save_output:
        df.to_csv(f'{cfg.Duration.output_path}/dur-output.csv', 
                  float_format='%0.5f', index=False)
        
    return df


def preliminary_checks(
    tr: obspy.Trace, 
    ptype: str, 
    cfg: types.ModuleType = config
) -> None:
    """Make sure all necessary information is present
    
    This function checks...
    1. The configuration file is setup correctly
    2. The trace has all relevant information
    3. There is sufficient time series data
    
    Args:
        tr: time series
        ptype: plot type
        cfg: configuration file
        
    Returns:
        None

    """
    
    ch.check_config(cfg.Duration())
    ch.check_waveform(tr)
    ch.check_plottype(ptype)       
        
    return None
            

def coda_duration(
    tr: obspy.Trace, 
    noise: float, 
    ptype: str, 
    cfg: types.ModuleType = config
) -> Tuple[float, float]:    
    """Measure duration and associate quality control
    
    Args:
        tr: time series
        noise: noise level
        ptype: plot type 
        cfg: configuration file
        
    Returns:
        dur: duration in seconds
        cc: correlation coefficient
    
    """

    dcfg = cfg.Duration()
    avg_time, avg_data_lin = moving_average(tr, dcfg)
    avg_data_log = log_envelope(avg_data_lin)
    fit_begin, fit_end = coda_fitting_window(tr, dcfg, avg_time, 
                                             avg_data_log, noise)
    coda_line = coda_line_fit(avg_time[fit_begin:fit_end], 
                              avg_data_log[fit_begin:fit_end])
    dur, i_dur = get_duration_index(dcfg, coda_line, noise)
    cc = get_correlation_coefficient(coda_line, avg_time, avg_data_log, 
                                     fit_begin, i_dur)
    
    if dcfg.plot:
        fig = plot.duration_plot(tr, avg_time, avg_data_lin, avg_data_log, 
                                 fit_begin, fit_end, dur, cc, coda_line, 
                                 noise, ptype, dcfg)
   
    return dur, cc


def moving_average(
    tr: obspy.Trace, 
    dcfg: config.Duration = config.Duration
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a moving average of the data
    
    Args:
        tr: time series
        dcfg: duration instance of configuration file
        
    Returns:
        averaged_time: moving average of time axis
        averaged_data: moving average of time series 
    
    """
    
    if dcfg.moving_average_window == 0:
        averaged_data = tr.data
        averaged_time = time_relative_p(tr)
    else:
        mov_avg_wind_samp = dcfg.moving_average_window * \
                            int(tr.stats.sampling_rate)
        time_wrt_p = time_relative_p(tr)
        time_averaging = pd.Series(time_wrt_p).rolling(
                         window=mov_avg_wind_samp).median()
        averaged_time = time_averaging.iloc[mov_avg_wind_samp-1:].values
        data_averaging = pd.Series(np.abs(hilbert(tr.data))).rolling(
                         window=mov_avg_wind_samp).mean()
        averaged_data = data_averaging.iloc[mov_avg_wind_samp-1:].values
    
    return averaged_time, averaged_data


def time_relative_p(tr: obspy.Trace) -> np.ndarray:    
    """Get time axis with respect to P- arrival
    
    Args: 
        tr: time series
        
    Returns:
        shifted time axis with respect to P-

    """
    
    p_wrt_b = (tr.stats.o - tr.stats.starttime) + tr.stats.p
    
    return np.arange(0, len(tr.data)) * tr.stats.delta - p_wrt_b


def log_envelope(data: np.ndarray) -> np.ndarray:    
    """Take log of coda envelope
    
    Args:
        data: coda envelope
        
    Returns
        log_env: the log (base 10) of the coda envelope
        
    """
    
    data_pos = np.where(data > 0, data, np.nan)
    log_env = np.log10(data_pos)
    log_env[np.isinf(log_env)] = np.nan
    log_env[np.isneginf(log_env)] = np.nan
    
    return log_env


def coda_fitting_window(
    tr: obspy.Trace, 
    dcfg: config.Duration, 
    time: np.ndarray, 
    data: np.ndarray, 
    noise: float
) -> Tuple[int, int]:  
    """Find coda search window by index
    
    Args:
        tr: time series
        dcfg: duration instance of configuration file
        time: averaged time axis
        data: averaged coda envelope data
        noise: noise level
        
    Returns:
        i_begin: search window beginning index
        i_end: search window ending index
        
    """
    
    search_window_begin = search_window(tr, time, dcfg, 'begin')
    search_window_end = search_window(tr, time, dcfg, 'end')
    i_max = np.where(data == np.nanmax(
            data[search_window_begin:search_window_end]))[0][0]
    i_end = fitting_window_end(tr, dcfg, data, i_max, noise)
    i_begin = fitting_window_start(dcfg, i_max, i_end)
    
    return i_begin, i_end


def search_window(
    tr: obspy.Trace, 
    time: np.ndarray, 
    dcfg: config.Duration, 
    position: str
) -> int:    
    """Find indices for coda search window
    
    Args:
        tr: time series
        time: time axis
        dcfg: duration instance of configuration file
        position: specify the part of the search window [begin, end]
        
    Returns:
        search_index: index value of desired search window position
    
    """
    
    search_seconds = search_window_seconds(tr, dcfg, position)
    search_index = np.where(np.sign(time - search_seconds) == 1)[0][0]
    
    return search_index


def search_window_seconds(
    tr: obspy.Trace, 
    dcfg: config.Duration, 
    position: str
) -> float:   
    """Get search window in terms of seconds
    
    Args:
        tr: time series
        dcfg: duration instance of configuration file
        position: specify the part of the search window [begin, end]
        
    Returns:
        seconds_wrt_p: time value of desired search window position
        
    Raises:
        AssertionError: if position is not begin or end
        
    """
    
    phase_p = phase_relative_p(tr, dcfg.signal_phase)
    
    assert position in ['begin', 'end'], \
        f'(ValueError) Position {position} unrecognized'
    if position == 'begin':
        seconds_wrt_p = phase_p + dcfg.signal_window_begin
    elif position == 'end':
        seconds_wrt_p = phase_p + dcfg.signal_window_end
    
    return seconds_wrt_p

        
def phase_relative_p(tr: obspy.Trace, phase: str) -> obspy.UTCDateTime:    
    """Get phase arrival in seconds relative to the P- arrival
    
    Args:
        tr: time series
        phase: the seismic phase
        
    Returns:
        phase_time: seconds format for phase        
    
    """
    
    if phase == 'O':
        phase_time = -tr.stats.p
    elif phase == 'P':
        phase_time = tr.stats.p - tr.stats.p
    elif phase == 'S':
        phase_time = tr.stats.s - tr.stats.p
        
    return phase_time


def fitting_window_start(
    dcfg: config.Duration, 
    i_max: int, 
    i_end: int
) -> int:   
    """Determine the start of the line fitting window
    
    Args:
        dcfg: duration instance of configuration file
        i_max: index of search window beginning
        i_end: index of search window end
        
    Returns:
        i_begin: index of fitting window beginning
        
    """
    
    if dcfg.start_fit_max > 1:
        i_begin = (i_end - i_max) / dcfg.start_fit_max + i_max
    else:
        i_begin = np.copy(i_max)
        
    return int(i_begin)


def fitting_window_end(
    tr: obspy.Trace, 
    dcfg: config.Duration, 
    data: np.ndarray, 
    i_max: int, 
    noise: float
) -> int:                                   
    """Find index of desired end-of-coda threshold
    
    Args: 
        tr: time series
        dcfg: duration instance of configuration file
        data: averaged coda envelope data
        i_max: index of search window beginning
        noise: noise level
        
    Returns:
        index of fitting window end
        
    Raises:
        AssertionError: if the fitting window is less than the s-p time
        
    """
    
    noise_diff = data[i_max:] - np.log10(dcfg.end_fit_noise * noise)
    i_noise = np.where(np.sign(noise_diff) == -1)[0]
    ch.check_fitting_window_end(i_noise, i_max, 
                                tr.stats.delta, tr.stats.s - tr.stats.p)
    
    return int(i_noise[0] + i_max)


def coda_line_fit(x0: np.ndarray, d0: np.ndarray) -> np.ndarray:    
    """Does least square inversion 
    
    Note...
    1. The slope is forced be positive
    2. Numpy is not allowed to parallelize in the background
    
    Args:
        x0: time axis
        d0: averaged data
        
    Returns:
        m.x: the parameters of the inversion
        
    Raises:
        AssertionError: if either array is not a float
        AssertionError: if time axis has NaN values        
        
    """
    
    x, d = ch.check_coda(x0, d0)
    ones = np.ones(np.shape(x))
    G = np.stack((ones, x), axis=-1)
    
    with threadpool_limits(limits=1, user_api='blas'):
        m = lsq(G, d, bounds=([-np.inf, -np.inf], [np.inf, 0]))
        
    return m.x


def get_duration_index(
    dcfg: config.Duration, 
    line: np.ndarray, 
    noise: float
) -> Tuple[float, int]:    
    """Find duration and duration index
    
    Args:
        dcfg: duration instance of configuration file
        line: line fit coefficients
        noise: noise level
        
    Returns:
        dur: duration in seconds
        i_dur: duration index
        
    """
    
    x_extended, y_extended = extend_line(line[1], line[0], 0., 10000., 0.001) 
    threshold = coda_line_end(dcfg, noise)
    cross = np.where(np.sign(y_extended - threshold) == -1)[0]
    ch.check_duration_index(cross)
    dur = x_extended[cross[0]]
    i_dur = cross[0]
    
    return dur, i_dur


def extend_line(
    m: float, 
    b: float, 
    xmin: float, 
    xmax: float, 
    xint: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Extends line given line parameters
    
    Args:
        m: slope
        b: intercept
        xmin: minimum of line extension
        xmax: minimum of line extension
        xint: interval of line extension
        
    Returns:
        x: extended x data
        y: extended y data
        
    """
    
    x = np.arange(xmin, xmax, xint)
    y = m * x + b
    
    return x, y                   
                    
    
def coda_line_end(dcfg: config.Duration, noise: float) -> float:   
    """Determine duration threshold
    
    Args:
        dcfg: duration instance of configuration file
        noise: noise level
        
    Returns:
        threshold: duration threshold
    
    """
    
    if dcfg.threshold_type == 'absolute':
        threshold = dcfg.duration_absolute_threshold
    elif dcfg.threshold_type == 'noise':
        threshold = dcfg.duration_noise_threshold * np.log10(noise)
        
    return threshold
 

def get_correlation_coefficient(
    line: np.ndarray, 
    time: np.ndarray, 
    data: np.ndarray, 
    data_begin: int, 
    line_end: int, 
    perc: float = 0.9
) -> np.ndarray:  
    """Format data to calculate correlation coefficient
    
    Args:
        line: line fit coefficients
        time: time axis
        data: averaged coda envelope data
        
    Returns:
        cc: correlation coefficient matrix
        
    Raises:
        UserWarning: if duration is longer than waveform segment
    
    """
    
    x_extended, y_extended = extend_line(line[1], line[0], 0, 10000, 0.001)
    line_begin = np.where(np.sign(x_extended - time[data_begin]) == 1)[0][0]
    data_end_0 = np.where(np.sign(time - x_extended[line_end]) == 1)[0]
    if len(data_end_0) == 0:
        data_end = int(np.around(perc * len(time)))
        warnings.warn(f'''Duration {x_extended[line_end]} 
                      longer than waveform  segment {time[-1]}. 
                      Fitting cc line to {perc * 100}% of waveform''')
    else:
        data_end = data_end_0[0]
    time_rounded = np.round(time, 3)
    _, i_line, i_data = np.intersect1d(x_extended[line_begin:line_end], 
                                       time_rounded[data_begin:data_end], 
                                       return_indices=True)
    x_data = time[i_data + data_begin]
    y_data = data[i_data + data_begin]
    x_line = x_extended[i_line + line_begin]
    y_line = y_extended[i_line + line_begin]
    xdata = np.column_stack((x_data, x_line))
    ydata = np.column_stack((10**y_data, 10**y_line))
    cc = calculate_cc(xdata, ydata)[0, 3]
    
    return cc
    
    
def calculate_cc(xdata: np.ndarray, ydata: np.ndarray) -> np.ndarray:   
    """Calculate Pearson correlation coefficient
    
    Args:
        xdata: waveform and line fit time axis
        ydata: waveform and line fit data axis
        
    Returns:
        cc: correlation matrix
        
    Raises:
        AssertionError: if correlation matrix is not real
        
    """
    
    cc = np.corrcoef(xdata, ydata, rowvar=False)
    ch.check_cc(cc, 0, 3)
    
    return cc   
       
    
def format_output(data: list) -> pd.DataFrame:
    """Turn list into dataframe
    
    Args:
        data: list of duration information
        
    Returns:
        df: dataframe of duration information
    
    Raises:
        AssertionError: if data size does not match column size
    
    """
    
    column_names = ['date', 'time', 'network', 'station', 'channel', 
                    'duration', 'cc', 'noise']
    assert len(data[0]) == len(column_names), \
        '(ValueError) Data length must match column length'
    df = pd.DataFrame(data, columns=column_names)
    
    return df