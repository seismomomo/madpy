"""
checks.py

performs checks from noise, amplitude, and duration modules
"""

import obspy
import types
import numpy as np
import pandas as pd 


def check_config(c):   
    """Verify config file is in order"""
    
    if (hasattr(c, 'noise_phase') and
        hasattr(c, 'noise_window_begin') and 
        hasattr(c, 'noise_window_end') and
        hasattr(c, 'signal_phase') and 
        hasattr(c, 'signal_window_begin') and 
        hasattr(c, 'signal_window_end') and
        hasattr(c, 'plot')):
        
        check_common_config(c)
        
        if hasattr(c, 'amp_factor'):
            
            check_amplitude_config(c)
            
        elif (hasattr(c, 'moving_average_window') and
            hasattr(c, 'start_fit_max') and
            hasattr(c, 'end_fit_noise') and 
            hasattr(c, 'threshold_type') and
            hasattr(c, 'duration_noise_threshold') and 
            hasattr(c, 'duration_absolute_threshold')):
            
            check_duration_config(c)
            
    else:
        
        raise AttributeError('''Missing attributes common to 
                             class Amplitude and class Duration''')
        
    return None
                    
                    
def check_common_config(c):                   
    """Check attributes common to class Amplitude and class Duration"""
    
    assert isinstance(c.noise_phase, str), \
        f'''(TypeError) 
        Phase {c.noise_phase} must be a string'''
    assert c.noise_phase in ['O', 'P', 'S'], \
        f'''(ValueError) 
        Phase {c.noise_phase} not recognized'''
    assert isinstance(c.noise_window_begin/1., float), \
        f'''(TypeError) 
        Noise window start {c.noise_window_begin} must be a float'''
    assert isinstance(c.noise_window_end/1., float), \
        f'''(TypeError) 
        Noise window end {c.noise_window_end} must be a float'''
    assert c.noise_window_end - c.noise_window_begin > 0, \
        '''(ValueError) 
        Noise window length must be greater than zero'''
    assert isinstance(c.signal_phase, str), \
        f'''(TypeError) 
        Phase {c.signal_phase} must be a string'''
    assert c.signal_phase in ['O', 'P', 'S'], \
        f'''(ValueError) 
        Phase {c.signal_phase} not recognized'''
    assert isinstance(c.signal_window_begin/1., float), \
        f'''(TypeError) 
        Signal window start {c.signal_window_begin} must be a float'''
    assert isinstance(c.signal_window_end/1., float), \
        f'''(TypeError) 
        Signal window end {c.signal_window_end} must be a float'''
    assert c.signal_window_end - c.signal_window_begin > 0, \
        f'''(ValueError) 
        Noise window length must be greater than zero'''
    assert isinstance(c.save_output, bool), \
        f'''(TypeError)
        Save output {c.save_output} must be a boolean'''
    assert isinstance(c.output_path, str), \
        f'''(TypeError)
        Output path {c.output_path} must be a string'''
    if c.save_output:
        assert len(c.output_path) > 0, \
            f'''(ValueError) 
            Output path must be specified if save_output=True'''
    assert isinstance(c.plot, bool), \
        f'''(TypeError) 
        Plot type {c.plot} must be a boolean'''
    assert isinstance(c.save_figure, bool), \
        f'''(TypeError) 
        Save figure {c.save_figure} must be a boolean'''
    assert isinstance(c.figure_path, str), \
        f'''(TypeError)
        Figure path {c.figure_path} must be a string'''
    if c.save_figure:
        assert c.plot, \
            f'''(ValueError)
            Plot must be generated if save_figure=True'''
        assert len(c.figure_path) > 0, \
            f'''(ValueError)
            Figure path must be specified if save_figure=True'''
    
    return None

def check_amplitude_config(c):
    """Check attributes for class Amplitude"""
    
    assert isinstance(c.amp_factor, float), \
        f'''(TypeError)
        Amplitude factor {c.amp_factor} must be a float'''
    assert c.amp_factor > 0, \
        f'''(ValueError)
        Amplitude factor {c.amp_factor} must be greater than 0'''
    
    return None

def check_duration_config(c):
    """Check attributes for class Duration"""
    
    assert isinstance(c.moving_average_window, int), \
        f'''(TypeError) 
        Moving average {c.moving_average_window} must be an integer'''
    assert c.moving_average_window >= 0, \
        f'''(ValueError) 
        Moving average {c.moving_average_window} must be at least zero'''
    assert isinstance(c.start_fit_max, int), \
        f'''(TypeError) 
        Coda fitting start {c.start_fit_max} must be an integer'''
    assert c.start_fit_max > 0, \
        f'''(ValueError) 
        Coda fitting start {c.start_fit_max} must be positive'''
    assert isinstance(c.end_fit_noise, float), \
        f'''(TypeError) 
        Coda fitting end {c.end_fit_noise} must be a float'''
    assert c.end_fit_noise > 0, \
        f'''(ValueError) 
        Coda fitting end {c.end_fit_noise} must be positive'''
    assert isinstance(c.threshold_type, str), \
        f'''(TypeError) 
        Coda fitting threshold {c.threshold_type} must be a string'''
    assert c.threshold_type in ['absolute', 'noise'], \
        f'''(ValueError) 
        Coda fitting threshold {c.threshold_type} not recognized'''
    assert isinstance(c.duration_noise_threshold, float), \
        f'''(TypeError) 
        Duration threshold {c.duration_noise_threshold} must be a float'''
    assert isinstance(c.duration_absolute_threshold, float), \
        f'''(TypeError) 
        Duration threshold {c.duration_absolute_threshold} must be a float'''
    
    return None


def check_waveform(tr):
    """Perform various checks on the Trace"""
    
    check_stats(tr)
    check_datagaps(tr)
    
    return None
    
    
def check_stats(tr):
    """Check Trace stats"""
    
    if (hasattr(tr.stats, 'o') and 
        hasattr(tr.stats, 'p') and 
        hasattr(tr.stats, 's')):
        
        assert isinstance(tr.stats.o, obspy.UTCDateTime), \
            f'''(TypeError) 
            Origin time {tr.stats.o} must be an Obspy datetime object'''
        assert tr.stats.o - tr.stats.starttime >= 0, \
            f'''(ValueError) 
            Origin time {tr.stats.o} must be later than 
            Trace start {tr.stats.starttime}'''
        assert isinstance(tr.stats.p, float), \
            f'''(TypeError) 
            P- arrival {tr.stats.p} must be a float'''
        assert tr.stats.p > 0, \
            f'''(ValueError) 
            P- wave traveltime {tr.stats.p} must be positive'''
        assert isinstance(tr.stats.s, float), \
            f'''(TypeError) 
            S- arrival {tr.stats.s} must be a float'''
        assert tr.stats.s > tr.stats.p, \
            f'''(ValueError) 
            S- wave traveltime {tr.stats.s} must be greater than 
            P- wave traveltime {tr.stats.p}'''
        
    else:
        
        raise AttributeError('Arrivals missing from the Trace object')
        
        
    return None


def check_datagaps(tr, nsec=100):
    """Verify there is sufficient data"""
    
    datalength = len(tr.data) * tr.stats.delta
    assert datalength > nsec, \
        f'''(ValueError) 
        Insufficient data length ({datalength} seconds)'''
        
    return None
    
    
def check_window(tr, starttime, endtime):
    """Verify there's enough data for window definitions"""
    
    starttime_window = starttime - tr.stats.starttime
    endtime_window = tr.stats.endtime - endtime
    
    assert starttime_window > 0, \
        f'''(ValueError) 
        Window start ({starttime}) is earlier than 
        Trace start ({tr.stats.starttime})'''
    assert endtime_window > 0, \
        f'''(ValueError) 
        Window end ({endtime}) is later than 
        Trace end ({tr.stats.endtime})'''
        
    return None


def check_amplitude(amp):
    """Verify the amplitude value is real"""
    
    if ((isinstance(amp, float) == False and 
         isinstance(amp, int) == False) or 
         isinstance(amp, bool) == True or 
         np.isneginf(amp) or 
         np.isinf(amp) or 
         np.isnan(amp) or 
         amp < 0):
        
        raise ValueError(f'Invalid amplitude value: {amp}')
        
    return None


def check_plottype(ptype):
    """Verify the plottype"""
    
    assert isinstance(ptype, str), \
        f'''(TypeError) 
        Plot type {ptype} must be a string'''
    assert ptype in ['linear', 'log'], \
        f'''(ValueError) 
        Plot option {ptype} not recognized'''
    
    return None


def check_fitting_window_end(i_end, i_max, dt, sp):
    """Verify coda end is real and appropriate"""
    
    assert len(i_end) > 0, \
        '''(ValueError) 
        Insufficient length for coda fitting'''
    assert (i_end[0] + i_max) * dt > sp, \
        '''(ValueError) 
        Insufficient length for coda fitting'''

    return None


def check_coda(x0, y0):
    """Verify data does not have NaN values"""
    
    assert x0.dtype == float, \
        '''(TypeError) 
        Time must be a float array'''
    assert y0.dtype == float, \
        '''(TypeError) 
        Data must be a float array'''
    assert len(x0[np.isnan(x0)]) == 0, \
        '''(ValueError) 
        Time cannot have NaN values'''
    
    if len(y0[np.isnan(y0)]) > 0:
        x = x0[~np.isnan(y0)]
        y = y0[~np.isnan(y0)]
    else:
        x = np.copy(x0)
        y = np.copy(y0)
        
    return x, y


def check_duration_index(cross):
    """Verify the duration location"""
    
    assert len(cross) > 0, \
        '''(ValueError) 
        Coda fit line does not cross noise threshold'''
    
    return None


def check_cc(cc, x, y):
    """Verify the correlation coefficient exists"""
    
    assert cc.dtype == float, \
        '''(ValueError) 
        Correlation matrix must be float'''
    assert len(cc) == 4, \
        '''(ValueError) 
        Invalid correlation coefficient'''
    assert len(cc[0]) == 4, \
        '''(ValueError) 
        Invalid correlation coefficient'''
    assert ~np.isnan(cc[x, y]), \
        '''(ValueError) 
        Invalid correlation coefficient'''
    assert ~np.isinf(cc[x, y]), \
        '''(ValueError) 
        Invalid correlation coefficient'''
    assert ~np.isneginf(cc[x, y]), \
        '''(ValueError) 
        Invalid correlation coefficient'''
    assert cc[x, y] != 0, \
        f'''(ValueError) 
        Invalid correlation coefficient: {cc[x, y]}'''
    assert cc[x, y] >= -1, \
        f'''(ValueError) 
        Invalid correlation coefficient: {cc[x, y]}'''
    assert cc[x, y] <= 1, \
        f'''(ValueError) 
        Invalid correlation coefficient: {cc[x, y]}'''
    
    return None