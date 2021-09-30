"""
noise.py

houses functions for noise processing

m. m. holt – august 2021
"""

# Installed imports
import obspy
import inspect
import numpy as np
import pandas as pd
from typing import Tuple

# Local imports
import madpy.config as config


def rms_noise(tr, measure):
    
    """
    trim waveform and measure noise level
    """
          
    tr_noise = trim_waveform_noise(tr.copy(), measure)
    noise = root_mean_square(tr_noise.data)
    
    # do the tests
    
    return noise

def trim_waveform_noise(tr, measure):
    
    """
    cut the waveform to noise window
    """
    
    starttime, endtime = noise_window(tr, measure)
    tr.trim(starttime=starttime, endtime=endtime)
    
    # check wave before return
    
    return tr

def noise_window(tr, measure):
    
    """
    discern amp/dur, p/s
    """
    
    # assert that segment is either signal or noise
    meas = measurement_type(measure)  

    arrival = arrival_time_utc(tr, meas.noise_phase)
    starttime = arrival + meas.noise_window_begin
    endtime = arrival + meas.noise_window_end
           
    return starttime, endtime
                                     
                    
def measurement_type(measure):
    
    """
    ampltiude or duration
    """
    
    # assert measure
    if measure == 'amplitude':
        meas = config.Amplitude()
    if measure == 'duration':
        meas = config.Duration()
        
    return meas
            
def arrival_time_utc(tr, phase):
    
    ############
    # assert name.upper() in [“O”, “P”, “S”], “Must be in …”
    if phase == 'O':
        arrival = tr.stats.o
    elif phase == 'P':
        arrival = tr.stats.o + tr.stats.p
    elif phase == 'S':
        arrival = tr.stats.o + tr.stats.s
        
    return arrival


def root_mean_square(data):
    
    """
    gets root mean square of numpy array
    """
    
    return np.sqrt(np.mean(data**2))

# def load_metadata(station_file: str) -> obspy.Inventory:
    
#     """
#     Load station metadata, travel time table, and event catalog for measurement
    
#     Args:
#         station_file (string): path to station metadata (XML file)
    
#     Returns:
#         inv: station inventory (obspy.Inventory)
        
#     Raises:
#         None 
#     """
    
#     # Load station info
#     inv = obspy.read_inventory(station_file, format='STATIONXML')
    
#     return inv


# def event_string(datetime: str) -> str:
    
#     """
#     Convert datetime to event string for saving
    
#     Args:
#         datetime (string): USGS format (YYYY-MM-DDTHH:MM:SS.MS)
        
#     Returns:
#         str: event string (YYYY.MM.DD.HH.MM.SS.MS)
        
#     Raises:
#         None
#     """
    
#     return f'{datetime[0:4]}.{datetime[5:7]}.{datetime[8:10]}.{datetime[11:13]}.{datetime[14:16]}.{datetime[17:19]}'


# def load_wf_data(path: str, event: str, components: str) -> obspy.Stream:
    
#     """
#     Load waveform data
    
#     Args:
#         path (str): path to wf data
#         event (str): event string
#         components (list): components 
        
#     Returns:
#         st: stream containing all data (obspy.Stream)
        
#     Raises:
#         None
#     """
    
#     st = obspy.Stream()
#     for component in components:
#         try:
#             st += obspy.read(f'{path}/{event}.{component}')
#         except FileNotFoundError:
#             pass
#     return st


# def get_station_metadata(
#     tr: obspy.Trace
# ) -> Tuple[str, str, str, str, float, float, float]:
    
#     """
#     Get station name and location
    
#     Args:
#         tr: waveform (obspy.Trace)
    
#     Returns:
#         net (str): network name
#         sta (str): station name
#         chan (str): channel gain information
#         comp (str): channel component
#         slat (float): station latitude in degrees
#         slon (float): station longitude in degrees
#         selev (float): station elevation in meters
        
#     Raises:
#         None
#     """
    
#     # Station name
#     net = tr.stats.network
#     sta = tr.stats.station
#     chan = tr.stats.channel[:-1]    
#     comp = tr.stats.channel[-1]
    
#     # Station location
#     slat = inv.select(station=sta, network=net)[0][0].latitude
#     slon = inv.select(station=sta, network=net)[0][0].longitude
#     selev = inv.select(station=sta, network=net)[0][0].elevation
    
#     return net, sta, chan, comp, slat, slon, selev
    
    
# def epi_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
#     """
#     Calculates epicentral distance between two coordinates assuming spherical earth
    
#     Args:
#         lat1 (float): latitude of location 1
#         lon1 (float): longitude of location 1
#         lat2 (float): latitude of location 2
#         lon2 (float): longitude of location 2
    
#     Returns:
#         d (float): epicentral distance in km
        
#     Raises:
#         None
#     """

#     # Convert to radians
#     lat1r = np.deg2rad(lat1)
#     lon1r = np.deg2rad(lon1)
#     lat2r = np.deg2rad(lat2)
#     lon2r = np.deg2rad(lon2)

#     # Calculate 
#     a = np.sin(np.divide(np.subtract(lat2r,lat1r),2))**2 + (
#         np.cos(lat1r)*np.cos(lat2r)*np.sin(np.divide(np.subtract(lon2r,lon1r),2))**2)
#     c = np.multiply(2,np.arctan2(np.sqrt(a),np.sqrt(1-a)))
#     d = 6371*c

#     return d


# def hypo_dist_km(
#     lat1: float, 
#     lon1: float, 
#     elev1: float, 
#     lat2: float, 
#     lon2: float, 
#     elev2: float
# ) -> float:
    
#     """
#     Calculates hypo central distance between two coordinates assuming spherical earth
#     Depths have negative elevation
    
#     Args:
#         lat1 (float): latitude of location 1
#         lon1 (float): longitude of location 1
#         elev1 (float): elevation of location 1
#         lat2 (float): latitude of location 2
#         lon2 (float): longitude of location 2
#         elev2 (float): elevation of location 2
    
#     Returns:
#         d (float): epicentral distance in km
        
#     Raises:
#         None
#     """
    

#     # Get different distances
#     epi_d = epi_dist_km(lat1, lon1, lat2, lon2)

#     # Hypocentral distance
#     dElev = np.abs(np.subtract(elev1, elev2))
#     d = np.sqrt(dElev**2 + epi_d**2)

#     return d


# def calculate_traveltimes(
#     df: pd.DataFrame, 
#     depth: float, 
#     distance: float
# ) -> Tuple[float, float]:
    
#     """
#     Calculate travel time from travel time table
    
#     Args:
#         df: travel time table (pd.DataFrame)
#         depth (float): depth of event 
#         dist (float): event-station distance
#         datum (float): correction for top of velocity model
        
#     Returns:
#         p: P-arrival in seconds relative to origin time
#         s: S-arrival in seconds relative to origin time
        
#     Raises:
#         None
#     """        
                                                    
#     new_depth = np.round(depth-DATUM, 1)
#     new_distance = np.round(distance, 0)
#     tt = df[ (df['depth'] == new_depth) & (df['dist'] == new_distance) ].values
#     p = tt[0][2]
#     s = tt[0][3]
    
#     return p, s


# def trim_waveform(tr0, otime, wave_segment):
    
#     # Create stream copy
#     tr = tr0.copy()
    
#     # Get wavesegment info
#     if wavesegment = 'noise':
#         starttime = otime + NOISE_WINDOW_BEGIN
#         endtime = otime + NOISE_WINDOW_END
#     else if wavesegment = 'signal':
#         starttime = otime + SIGNAL_WINDOW_BEGIN
#         endtime = otime + SIGNAL_WINDOW_END
#     else # Exception:
#         pass # Add Exception

#     # Trim waveform
#     tr.trim(starttime=starttime, endtime=endtime)

#     return tr


# def rms_noise(tr):
    
#     """
#     Gets the RMS of the noise
#     """

#     ## Extract data
#     dat = tr.data*1e3

#     ## Noise level
#     noise = np.sqrt(np.mean(dat**2))

#     return noise


# def max_peak_to_peak(data):
    
#     # Get peak-to-peak amplitudes
#     inflection_points = get_inflection_points(data)
#     amplitudes = np.diff(inflection_points)
#     amp = np.max(np.abs(amplitudes))/2
    
#     # check for bad amplitudes
    
#     # Get indices for peak-to-peak amplitudes
#     idx = get_peak_to_peak_indices(amplitudes)
    
#     return amp, idx


# def inflection_points(data):
    
#     nan_points_10000 = np.concatenate([[0], (np.diff(np.sign(np.diff(data))) == 0)*10000, [0]])
#     nan_points = np.add(data, nan_points_10000)
#     nan_points[nan_points > 1000] = np.nan
#     inflection_points = nan_points[~np.isnan(nan_points)]
    
    
# def get_peak_to_peak_indices(amplitudes):
    
#     # Find possible indices of max peak-to-peak
#     i_diff = np.where(np.abs(amplitudes) == np.max(np.abs(amplitudes)))
#     i_peak1 = i_diff[0][0]
#     i_peak2 = i_peak1 + 1
#     peak1 = peaks[i_peak1]
#     peak2 = peaks[i_peak2]
#     i_p1_0 = np.where(nan_points == peak1)
#     i_p2_0 = np.where(nan_points == peak2)
    
#     # Find closest indices to max peak-to-peak
#     if len(i_p1[0]) > 1 or len(i_p2[0]) > 1:
#         x_p1 = np.repeat(i_p1_0[0], len(i_p2_0[0]))
#         x_p2 = np.tile(i_p2_0[0], len(i_p2_0[0]))
#         x_p2_p1 = np.subtract(x_p2, x_p1)
#         x_p2_p1 = np.divide(x_p2_p1, 1.0)
#         x_p2_p1[x_p2_p1 < 0] = np.nan
#         i_x = np.where(x_p2_p1 == np.nanmin(x_p2_p1))
#         i_p1 = x_p1[int(i_x[0])]
#         i_p2 = x_p2[int(i_x[0])]
#         idx = np.array([i_p1, i_p2])
#     else:
#         i_p1 = i_p1_0[0]
#         i_p2 = i_p2_0[0]
#         idx = np.array([i_p1, i_p2])

#     return ind