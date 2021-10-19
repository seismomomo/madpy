"""
generates the data
"""

import obspy
import numpy as np

def main():
    
    """
    generate test data
    """
    
    # General data
    npts = 30000
    dt = 0.01
    o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
    tr = generate_trace(npts, dt, o)
    
    # Amplitude data
    amp_data = amplitude_data(npts)
    tr_amp = tr.copy()
    tr_amp.data = amp_data
    tr_amp.write('amplitude.mseed', format='MSEED')
    
    # Duration data
    dur_data = duration_data(npts)
    tr_dur = tr.copy()
    tr_dur.data = dur_data
    tr_dur.write('duration.mseed', format='MSEED')
    
    
def duration_data(npts):
    
    """
    generate fake duration data
    """
    
    mean = 0
    stdev = 0.1
    m_coda, b_coda = -0.00025, -4.625
    m_p, b_p = 0.00267, -20.68
    p1, p2 = 4000, 5500
    c1, c2 = 5500, 21500
    base = -10
    log_data = np.zeros((npts + 1,)) + base
    log_data[p1:p2] = m_p * np.arange(p1, p2) + b_p
    log_data[c1:c2] = m_coda * np.arange(c1, c2) + b_coda
    log_data += np.random.normal(mean, stdev, npts + 1)
    lin_data = 10 ** log_data
    idx = np.random.choice(np.arange(len(lin_data)), int(0.5 * len(lin_data)))
    lin_data[idx] *= -1
    
    
    return lin_data
    
    
def amplitude_data(npts):
    
    """
    generate fake amplitude data
    """
    
    mean = 0
    stdev = 0.1
    p1, p2, pn = -1.5, 1.7, 10
    id1 = 5995
    data = np.random.normal(mean, stdev, npts + 1)
    data[id1:id1 + pn] = np.linspace(p1, p2, pn)
    
    return data    
    

def generate_trace(npts, dt, o):
    
    """
    This populates the trace
    """
    
    tr = obspy.Trace()
    tr.stats.network = 'IL'
    tr.stats.station = 'UIC'
    tr.stats.channel = 'SES'
    tr.stats.starttime = o - 30
    tr.stats.sampling_rate = 1/dt
    tr.stats.delta = dt
    tr.stats.npts = npts
    
    return tr
    
    

if __name__ == '__main__':
    main()

