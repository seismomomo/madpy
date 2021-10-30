import obspy
import unittest
import numpy as np
import madpy.noise as n
import madpy.tests.testdata.config as cfg


class TestNoise(unittest.TestCase):
    
    
    def test_rms_noise(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            tr.stats.p = 10.
            tr.stats.s = 20.
 
        self.assertAlmostEqual(n.rms_noise(st[0], 'amplitude', cfg), 0.1, 1)
        self.assertAlmostEqual(n.rms_noise(st[0], 'duration', cfg), 0.1, 1)
        self.assertAlmostEqual(np.log10(n.rms_noise(st[1], 'amplitude')), -10, 0)
        self.assertAlmostEqual(np.log10(n.rms_noise(st[1], 'duration')), -10, 0)
        
        
    def test_trim_waveform_noise(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            tr.stats.p = 10.
            tr.stats.s = 20.
            tra = n.trim_waveform_noise(tr.copy(), 'amplitude', cfg)
            trd = n.trim_waveform_noise(tr.copy(), 'duration', cfg)
            self.assertEqual(len(tra.data), 2901)
            self.assertEqual(len(trd.data), 1001)
            
            
    def test_noise_window(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            tr.stats.p = 10.
            tr.stats.s = 20.
            starttime_amp, endtime_amp = n.noise_window(tr, 'amplitude', cfg)
            starttime_dur, endtime_dur = n.noise_window(tr, 'duration', cfg)
            noise_begin_amp = obspy.UTCDateTime('2020-10-10T13:04:40.00')
            noise_end_amp = obspy.UTCDateTime('2020-10-10T13:05:09.00')
            noise_begin_dur = obspy.UTCDateTime('2020-10-10T13:04:40.00')
            noise_end_dur = obspy.UTCDateTime('2020-10-10T13:04:50.00')
            self.assertEqual(starttime_amp, noise_begin_amp)
            self.assertEqual(endtime_amp, noise_end_amp)
            self.assertEqual(starttime_dur, noise_begin_dur)
            self.assertEqual(endtime_dur, noise_end_dur)
            
    def test_measurement_type(self):
        
        self.assertRaises(AssertionError, n.measurement_type, 'option')
        self.assertEqual(n.measurement_type('amplitude', cfg).noise_phase, 
                         cfg.Amplitude.noise_phase)
        self.assertEqual(n.measurement_type('duration', cfg).noise_phase, 
                         cfg.Duration.noise_phase)
                
        
    def test_arrival_time_utc(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            tr.stats.p = 10.
            tr.stats.s = 20.
            p = obspy.UTCDateTime('2020-10-10T13:05:10.00')
            s = obspy.UTCDateTime('2020-10-10T13:05:20.00')
            self.assertEqual(n.arrival_time_utc(tr, 'O'), tr.stats.o)
            self.assertEqual(n.arrival_time_utc(tr, 'P'), p)
            self.assertEqual(n.arrival_time_utc(tr, 'S'), s)
            
    
    def test_root_mean_square(self):
        
        data = np.cos(np.arange(-np.pi, np.pi, np.pi / 12))
        self.assertAlmostEqual(n.root_mean_square(data), 1 / np.sqrt(2))
        data = np.ones((10,)) * 5
        self.assertEqual(n.root_mean_square(data), 5)
        
        
        
if __name__ == '__main__':
    unittest.main()