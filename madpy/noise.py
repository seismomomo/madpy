"""
noise.py

measure the time series noise level
"""

# Installed 
import obspy
import types
import numpy as np
import pandas as pd
from typing import Tuple

# Local 
import madpy.checks as ch
import madpy.config as config


def rms_noise(
    tr: obspy.Trace, 
    measure: str, 
    cfg: types.ModuleType = config
) -> float:
    """Measure the noise level of the time series
    
    Args:
        tr: time series 
        measure: measurement type [amplitude, duration]
        cfg: configuration file
    
    Returns:
        noise: the time series noise level
        
    """
    
    tr_noise = trim_waveform_noise(tr.copy(), measure, cfg)
    noise = root_mean_square(tr_noise.data)
    
    return noise


def trim_waveform_noise(
    tr: obspy.Trace, 
    measure: str, 
    cfg: types.ModuleType = config
) -> obspy.Trace:  
    """Cut the time series to noise window
    
    Args:
        tr: time series
        measure: measurement type [amplitude, duration]
        cfg: configuration file
        
    Returns:
        tr: trimmed time series
        
    """
    
    starttime, endtime = noise_window(tr, measure, cfg)
    tr.trim(starttime=starttime, endtime=endtime)
        
    return tr


def noise_window(
    tr: obspy.Trace, 
    measure: str, 
    cfg: types.ModuleType = config
) -> Tuple[obspy.UTCDateTime, obspy.UTCDateTime]:    
    """Get the starttimes and endtimes of the noise window
    
    Args:
        tr: time series
        measure: measurement type [amplitude, duration]
        cfg: configuration file
        
    Returns:
        starttime: signal window beginning date
        endtime: signal window ending date
    
    Raises:
        AssertionError: Window begins before time series begins
        AssertionError: Window ends after time series ends
        
    """
    
    c = measurement_type(measure, cfg)  

    arrival = arrival_time_utc(tr, c.noise_phase)
    starttime = arrival + c.noise_window_begin
    endtime = arrival + c.noise_window_end
    ch.check_window(tr, starttime, endtime)
           
    return starttime, endtime
                                     
                    
def measurement_type(
    measure: str, 
    cfg: types.ModuleType = config
) -> config.Amplitude:    
    """Choose the appropriate class for measurement type
    Args:
        measure: measurement type [amplitude, duration]
        cfg: configuration file

    Returns:
        c: class instance            

    Raises:
        AssertionError: if measurement type is not amplitude or duration
            
    """
    
    assert measure in ['amplitude', 'duration']
    if measure == 'amplitude':
        c = cfg.Amplitude()
    if measure == 'duration':
        c = cfg.Duration()
    
    return c


def arrival_time_utc(tr: obspy.Trace, phase: str) -> obspy.UTCDateTime:   
    """Change arrival times to date format
    
    Args: 
        tr: time series
        phase: the seismic phase

    Returns:
        arrival: datetime format for phase
            
    """
   
    if phase == 'O':
        arrival = tr.stats.o
    elif phase == 'P':
        arrival = tr.stats.o + tr.stats.p
    elif phase == 'S':
        arrival = tr.stats.o + tr.stats.s
        
    return arrival


def root_mean_square(data: np.ndarray) -> float:
    """Calculate root mean square of time series
    
    Args:
        data: numpy array
    
    Returns:
        RMS value of the data

    """
    
    return np.sqrt(np.nanmean(data ** 2))