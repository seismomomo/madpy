import obspy
import unittest
import numpy as np
import madpy.checks as ch
import madpy.tests.testdata.config as cfg


class TestChecks(unittest.TestCase):
            
    def test_check_config(self):
        
        class Measurements: pass
        self.assertRaises(AttributeError, ch.check_config, Measurements())
        self.assertIsNone(ch.check_config(cfg.Amplitude()))
        cfg.Amplitude.noise_phase = 'Pn'
        self.assertRaises(AssertionError, ch.check_config, cfg.Amplitude())
        cfg.Amplitude.noise_phase = 'P'
        cfg.Amplitude.amp_factor = -2.
        self.assertRaises(AssertionError, ch.check_config, cfg.Amplitude())
        cfg.Amplitude.plot = 'Yes'
        self.assertRaises(AssertionError, ch.check_config, cfg.Amplitude())
        cfg.Amplitude.plot = False
        cfg.Amplitude.signal_window_begin = 50.
        self.assertRaises(AssertionError, ch.check_config, cfg.Amplitude())
        cfg.Amplitude.signal_window_begin = -1
        cfg.Amplitude.save_figure = True
        self.assertRaises(AssertionError, ch.check_config, cfg.Amplitude())
        cfg.Amplitude.save_figure = False
        
        self.assertIsNone(ch.check_config(cfg.Duration()))
        cfg.Duration.signal_phase = 'Sg'
        self.assertRaises(AssertionError, ch.check_config, cfg.Duration())
        cfg.Duration.signal_phase = 'S'
        cfg.Duration.moving_average_window = -2
        self.assertRaises(AssertionError, ch.check_config, cfg.Duration())
        cfg.moving_average_window = 2
        cfg.threshold_type = 'pre-p noise'
        self.assertRaises(AssertionError, ch.check_config, cfg.Duration())
        cfg.threshold_type = 'noise'
        cfg.plot = True
        cfg.save_figure = True
        cfg.figure_path = ''
        self.assertRaises(AssertionError, ch.check_config, cfg.Duration())
        cfg.plot = False
        cfg.save_figure = False

        
    def test_check_waveform(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            tr.stats.p = 10.
            self.assertRaises(AttributeError, ch.check_config, tr)
            tr.stats.s = 20.
            self.assertIsNone(ch.check_stats(tr))
            tr.stats.o = '2020-10-10T13:05:00.00'
            self.assertRaises(AssertionError, ch.check_stats, tr)
            tr.stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        
        
    def test_check_datagaps(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            self.assertIsNone(ch.check_datagaps(tr))
            n = int(len(tr.data) * 0.25)
            tr.data = tr.data[0:n]
            self.assertRaises(AssertionError, ch.check_datagaps, tr) 
        
    
    def test_check_window(self):
        
        st = obspy.read('testdata/*.mseed')
        for tr in st:
            starttime = obspy.UTCDateTime('2020-10-10T13:05:00.00')
            endtime = obspy.UTCDateTime('2020-10-10T13:07:00.00')
            self.assertIsNone(ch.check_window(tr, starttime, endtime))
            starttime = obspy.UTCDateTime('2020-10-10T13:04:00.00')
            self.assertRaises(AssertionError, ch.check_window, tr, starttime, endtime)
            endtime = obspy.UTCDateTime('2020-10-10T13:08:00.00')
            self.assertRaises(AssertionError, ch.check_window, tr, starttime, endtime)
        
        
    def test_check_amplitude(self):
        
        self.assertIsNone(ch.check_amplitude(0.5))
        self.assertRaises(ValueError, ch.check_amplitude, np.nan)
        self.assertRaises(ValueError, ch.check_amplitude, np.inf)
        self.assertRaises(ValueError, ch.check_amplitude, -np.inf)
        self.assertRaises(ValueError, ch.check_amplitude, -0.5)
        self.assertRaises(ValueError, ch.check_amplitude, None)
        self.assertRaises(ValueError, ch.check_amplitude, True)
        self.assertRaises(ValueError, ch.check_amplitude, {'test': 'dict'})
        self.assertRaises(ValueError, ch.check_amplitude, ['list', 5])        
        
        
    def test_check_fitting_window_end(self):
        
        i_max0 = 20000
        i_end0 = np.arange(500, 5005)
        dt = 0.01
        sp = 10
        self.assertIsNone(ch.check_fitting_window_end(i_end0, i_max0, dt, sp))
        i_end1 = []
        self.assertRaises(AssertionError, ch.check_fitting_window_end, i_end1, i_max0, dt, sp)
        i_max1 = 2
        self.assertRaises(AssertionError, ch.check_fitting_window_end, i_end0, i_max1, dt, sp)
    
    
    def test_check_plottype(self):
        
        self.assertIsNone(ch.check_plottype('linear'))
        self.assertIsNone(ch.check_plottype('log'))
        self.assertRaises(AssertionError, ch.check_plottype, 2)
        self.assertRaises(AssertionError, ch.check_plottype, 'fourier')
        
        
    def test_check_duration_index(self):
        
        cross = np.arange(0, 10, dtype=float)
        self.assertIsNone(ch.check_duration_index(cross))
        self.assertRaises(AssertionError, ch.check_duration_index, [])      
    
    
    def test_check_cc(self):
        
        cc = np.array([
            [0.1, 0.8, 0.5, 0.9],
            [0.9, 0.1, 0.8, 0.5],
            [0.5, 0.9, 0.1, 0.8],
            [0.8, 0.5, 0.9, 0.1]
        ])
        self.assertIsNone(ch.check_cc(cc, 1, 2))
        self.assertRaises(AssertionError, ch.check_cc, cc.astype(int), 1, 2)
        self.assertRaises(AssertionError, ch.check_cc, cc[0:3, :], 1, 2)
        self.assertRaises(AssertionError, ch.check_cc, cc[:, 0:3], 1, 2)
        cc[1, 2] = np.nan
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        cc[1, 2] = np.inf
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        cc[1, 2] = -np.inf
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        cc[1, 2] = 0
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        cc[1, 2] = -10
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        cc[1, 2] = 25
        self.assertRaises(AssertionError, ch.check_cc, cc, 1, 2)
        
        
    def test_check_coda(self):
        
        x0 = np.arange(0, 100)
        y0 = np.arange(0, 100)
        x1 = np.arange(0, 100, dtype=float)
        y1 = np.arange(0, 100, dtype=float)
        self.assertRaises(AssertionError, ch.check_coda, x0, y1)
        self.assertRaises(AssertionError, ch.check_coda, x1, y0)
        self.assertRaises(AssertionError, ch.check_coda, x0, y0)
        x2, y2 = ch.check_coda(x1, y1)
        self.assertEqual(len(x2), 100)
        self.assertEqual(len(y2), 100)
        x2[5:10] = np.nan
        self.assertRaises(AssertionError, ch.check_coda, x2, y1)
        y2[60:72] = np.nan
        x3, y3 = ch.check_coda(x1, y2)
        self.assertEqual(len(x3), 88)
        self.assertEqual(len(y3), 88)
        
                
if __name__ == '__main__':
    unittest.main()