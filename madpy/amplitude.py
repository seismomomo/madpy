"""
amplitude.py

measure the maximum peak-to-peak amplitude
"""

import obspy
import types
import numpy as np
import pandas as pd
import madpy.noise as n
from typing import Tuple
import madpy.checks as ch
import madpy.config as config
import matplotlib.pyplot as plt
import madpy.plotting.amp as plot


def measure_amplitude(
    st: obspy.Stream, 
    cfg: types.ModuleType = config, 
) -> pd.DataFrame:
    """Measure noise level and amplitude

    Args:
        st: stream containing one or more time series
        cfg: configuration file
        
    Returns:
        df: dataframe of time series amplitude information

    """
    
    output = []
    for tr in st:
    
        preliminary_checks(tr, cfg)
        noise = n.rms_noise(tr, 'amplitude', cfg)
        amplitude = max_amplitude(tr, noise, cfg)
        output.append([str(tr.stats.o.date), str(tr.stats.o.time)[:11],
                       tr.stats.network, tr.stats.station, tr.stats.channel,
                       amplitude, noise])
        
    df = format_output(output)
    if cfg.Amplitude.save_output:
        df.to_csv(f'{cfg.Amplitude.output_path}/amp-output.csv', 
                  float_format='%0.5f', index=False)
        
    return df


def preliminary_checks(
    tr: obspy.Trace,  
    cfg: types.ModuleType = config
) -> None:
    """Make sure all necessary information is present
    
    This function checks...
    1. The configuration file is setup correctly
    2. The trace has all relevant information
    3. There is sufficient time series data
    
    Args:
        tr: time series
        cfg: configuration file
        
    Returns:
        None

    """
    
    ch.check_config(cfg.Amplitude())
    ch.check_waveform(tr)
        
    return None

        
def max_amplitude(
    tr: obspy.Trace, 
    noise: float, 
    cfg: types.ModuleType = config
) -> float:
    """Measure maximum peak-to-peak amplitude
    
    Args:
        tr: time series
        noise: noise level
        cfg: configuration file
    
    Returns: 
        amp: maximum peak-to-peak amplitude
        
    Raises:
        ValueError: if max amplitude is not real and positive
    
    """
    
    acfg = cfg.Amplitude()
    tr_signal = trim_waveform_signal(tr.copy())
    peaks_nan = inflection_points(tr_signal.data)
    peaks = remove_nan(peaks_nan)
    p2p_amplitudes = np.diff(peaks)
    amp = np.max(np.abs(p2p_amplitudes)) * acfg.amp_factor
    ch.check_amplitude(amp)
    
    if acfg.plot:
        indices = p2p_indices(tr_signal, peaks, p2p_amplitudes)
        plot.amplitude_plot(tr, tr_signal, amp, indices, noise, acfg)
        
    return amp


def trim_waveform_signal(
    tr: obspy.Trace, 
    cfg: types.ModuleType = config
) -> obspy.Trace:
    """Cut the time series to signal window
    
    Args:
        tr: time series
        cfg: configuration file
        
    Returns:
        tr: trimmed time series
    
    """
    
    starttime, endtime = signal_window(tr, cfg)
    tr.trim(starttime=starttime, endtime=endtime)
    
    return tr

    
def signal_window(
    tr: obspy.Trace, 
    cfg: types.ModuleType = config
) -> Tuple[obspy.UTCDateTime, obspy.UTCDateTime]:    
    """Get the starttimes and endtimes of signal window
    
    Args:
        tr: time series
        cfg: configuration file
        
    Returns:
        starttime: signal window beginning date
        endtime: signal window ending date
        
    Raises:
        AssertionError: Window begins before time series begins
        AssertionError: Window ends after time series ends
        
    """
    
    acfg = cfg.Amplitude()
    arrival = n.arrival_time_utc(tr, acfg.signal_phase)
    starttime = arrival + acfg.signal_window_begin
    endtime = arrival + acfg.signal_window_end
    ch.check_window(tr, starttime, endtime)
    
    return starttime, endtime


def inflection_points(data: np.ndarray) -> np.ndarray:    
    """Isolate the peaks of an array
    
    Args:
        data: time series
        
    Returns:
        inflection_points: peaks of the time series
    
    """
    
    nan_points = np.concatenate([[0], np.diff(np.sign(np.diff(data))), [0]])
    nan_points[nan_points == 0] = np.nan
    nan_points[~np.isnan(nan_points)] = 0
    inflection_points = nan_points + data
    
    return inflection_points


def remove_nan(array: np.ndarray) -> np.ndarray:   
    """Remove NaN values
    
    Args:
        array: time series
        
    Returns:
        the time series without NaN values
        
    """
    
    return array[~np.isnan(array)]


def p2p_indices(
    tr: obspy.Trace, 
    peaks: np.ndarray, 
    amplitudes: np.ndarray
) -> Tuple[float, float]:
    """Get peak indices of max peak-to-peak amplitude
    
    Args:
        tr: time series
        peaks: the inflection points of the time series
        amplitudes: values of each peak to peak
        
    Return:
        idx: indices of two peaks associated with maximum amplitude
        
    """
    
    i_diff = np.where(np.abs(amplitudes) == np.max(np.abs(amplitudes)))
    i_peak1 = i_diff[0][0]
    i_peak2 = i_peak1 + 1
    peak1 = peaks[i_peak1]
    peak2 = peaks[i_peak2]
    nan_points = inflection_points(tr.data)
    i_p1_0 = np.where(nan_points == peak1)
    i_p2_0 = np.where(nan_points == peak2)
    idx = p2p_indices_check(i_p1_0, i_p2_0)
    
    return idx


def p2p_indices_check(i_p1_0: float, i_p2_0: float) -> Tuple[float, float]:
    """Verify the indices are associated with the peaks
    
    Args:
        i_p1_0: preliminary peak 1
        i_p2_0: preliminary peak 2
        
    Returns:
        idx: final peaks
        
    """
    
    if len(i_p1_0[0]) > 1 or len(i_p2_0[0]) > 1:
        x_p1 = np.repeat(i_p1_0[0], len(i_p2_0[0]))
        x_p2 = np.tile(i_p2_0[0], len(i_p2_0[0]))
        x_p2_p1 = np.subtract(x_p2, x_p1)
        x_p2_p1 = np.divide(x_p2_p1, 1.0)
        x_p2_p1[x_p2_p1 < 0] = np.nan
        i_x = np.where(x_p2_p1 == np.nanmin(x_p2_p1))
        i_p1 = x_p1[int(i_x[0])]
        i_p2 = x_p2[int(i_x[0])]
        idx = np.array([i_p1, i_p2])
    else:
        i_p1 = i_p1_0[0]
        i_p2 = i_p2_0[0]
        idx = np.array([i_p1, i_p2])
        
    return idx


def format_output(data: list) -> pd.DataFrame:    
    """Turn list into dataframe
    
    Args:
        data: list of amplitude information
        
    Returns:
        df: dataframe of amplitude information
    
    Raises:
        AssertionError: if data size does not match column size
    
    """
    
    column_names = ['date', 'time', 'network', 'station', 'channel', 
                    'amplitude', 'noise']
    assert len(data[0]) == len(column_names), \
        '(ValueError) Data length must match column length'
    df = pd.DataFrame(data, columns=column_names)
    
    return df