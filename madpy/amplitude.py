"""
amplitude.py

This script measures the maximum peak-to-peak amplitude

m.m. holt – september 2021
"""

# Installed imports
import obspy
import numpy as np
import pandas as pd

# Local imports
import madpy.config as config
import madpy.plotting as plot
import madpy.processing as proc


def measure_amplitude(st, save_output=False, outfile=None):
    
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
        noise = proc.rms_noise(tr, 'amplitude')
        
        # Measure amplitude
        amplitude = max_amplitude(tr, noise)
        
        # Add to data #np.sqrt((amplitude/snr)**2)
        output.append([str(tr.stats.o.date), str(tr.stats.o.time)[:11],
                       tr.stats.elat, tr.stats.elon, tr.stats.edep,
                       tr.stats.station, tr.stats.channel,
                       tr.stats.slat, tr.stats.slon, tr.stats.selev,
                       amplitude, np.divide(amplitude, noise)])
        
    # Format output
    df = format_output(output)
    
    # Save if desired
    if save_output:
        df.to_csv(f'{outfile}', float_format='%0.5f', index=False)
        
    return df

        
def max_amplitude(tr, noise):
    
    """
    trim waveform and measure max peak-to-peak
    """
    
    tr_signal = trim_waveform_signal(tr.copy())
    peaks,_ = inflection_points(tr_signal.data)
    p2p_amplitudes = np.diff(peaks)
    amp = np.max(np.abs(p2p_amplitudes)) / 2
    
    # check for bad amplitude
    
    # plot
    cfg = config.Amplitude()
    if cfg.amp_plot == 'yes':
        indices = p2p_indices(tr_signal, peaks, p2p_amplitudes)
        plot.amplitude_plot(tr, tr_signal, amp, indices, noise)      
        
   
    return amp


def trim_waveform_signal(tr):
    
    """
    cut just signal portion
    """
    
    starttime, endtime = signal_window(tr)
    tr.trim(starttime=starttime, endtime=endtime)
    # check wave before return
    
    return tr

    
def signal_window(tr):
    
    amp = config.Amplitude()
    arrival = proc.arrival_time_utc(tr, amp.signal_phase)
    starttime = arrival + amp.signal_window_begin
    endtime = arrival + amp.signal_window_end
    
    return starttime, endtime


def inflection_points(data):
    
    """
    isolate only peaks
    """
    
    nan_points = np.concatenate([[0], np.diff(np.sign(np.diff(data))), [0]])
    nan_points[nan_points == 0] = np.nan
    nan_points[~np.isnan(nan_points)] = 0
    inflection_points = nan_points + data
    
    return inflection_points[~np.isnan(inflection_points)], inflection_points


def p2p_indices(tr, peaks, amplitudes):
    
    # Find possible indices of max peak-to-peak
    i_diff = np.where(np.abs(amplitudes) == np.max(np.abs(amplitudes)))
    i_peak1 = i_diff[0][0]
    i_peak2 = i_peak1 + 1
    peak1 = peaks[i_peak1]
    peak2 = peaks[i_peak2]
    _, nan_points = inflection_points(tr.data)
    i_p1_0 = np.where(nan_points == peak1)
    i_p2_0 = np.where(nan_points == peak2)
    
    # Find closest indices to max peak-to-peak
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


def format_output(data):
    
    """
    turn list into dataframe
    """
    
    column_names = ['date', 'time', 'event_lat', 'event_lon', 'event_depth',
                    'station', 'channel', 'station_lat', 'station_lon', 'station_elev',
                    'amplitude', 'snr']
    
    df = pd.DataFrame(data, columns=column_names)
    
    return df