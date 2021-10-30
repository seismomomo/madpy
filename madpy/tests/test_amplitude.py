import obspy
import unittest
import numpy as np
import madpy.amplitude as amp
import madpy.tests.testdata.config as cfg


class TestAmplitude(unittest.TestCase):
    
    
    def test_measure_amplitude(self):
        
        st = obspy.read('testdata/amplitude.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        df_amp = amp.measure_amplitude(st, cfg)
        self.assertEqual(len(df_amp), len(st))
        self.assertEqual(len(df_amp.columns), 7)
        
    
    def test_max_amplitude(self):
        
        st = obspy.read('testdata/amplitude.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        self.assertEqual(amp.max_amplitude(st[0], 0.1, cfg), 1.6)
        
        
    def test_trim_waveform_signal(self):
        
        st = obspy.read('testdata/amplitude.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        tr = amp.trim_waveform_signal(st[0], cfg)
        self.assertEqual(len(tr.data), 3101)
        
        
    def test_signal_window(self):
        
        st = obspy.read('testdata/amplitude.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        starttime, endtime = amp.signal_window(st[0], cfg)
        signal_begin = obspy.UTCDateTime('2020-10-10T13:05:19.00')
        signal_end = obspy.UTCDateTime('2020-10-10T13:05:50.00')
        self.assertEqual(starttime, signal_begin)
        self.assertEqual(endtime, signal_end)
        
        
    def test_inflection_points(self):
        
        data = np.array([0, -1, 0, 1, 0.5, -0.5, -1, -0.8, 0])
        data_ip = np.array([np.nan, -1, np.nan, 1, np.nan, np.nan, -1, np.nan, np.nan])
        ip = amp.inflection_points(data)
        self.assertEqual(len(ip[np.isnan(ip)]), 6)
        self.assertEqual(np.nansum(data_ip - ip), 0)
        
        
    def test_remove_nan(self):
        
        ip_nan = np.array([np.nan, -1, np.nan, 1, np.nan, np.nan, -1, np.nan, np.nan])
        data_ip = np.array([-1, 1, -1])
        ip = amp.remove_nan(data_ip)
        self.assertEqual(len(ip), 3)
        self.assertEqual(np.sum(data_ip - ip), 0)
        
        
    def test_p2p_indices(self):
        
        st = obspy.read('testdata/amplitude.mseed')
        starttime = obspy.UTCDateTime('2020-10-10T13:05:19.00')
        endtime = obspy.UTCDateTime('2020-10-10T13:05:50.00')
        st[0].trim(starttime=starttime, endtime=endtime)
        peaks_nan = amp.inflection_points(st[0].data.copy())
        peaks = amp.remove_nan(peaks_nan)
        idx = amp.p2p_indices(st[0].data.copy(), peaks, np.diff(peaks))
        self.assertEqual(idx[0], 1095)
        self.assertEqual(idx[1], 1104)
        
        
    def test_format_output(self):
        
        data = ['2020', '13:00:00', 1., 2., 3.]
        self.assertRaises(AssertionError, amp.format_output, data)
        

    
if __name__ == '__main__':
    unittest.main()