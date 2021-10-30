import obspy
import unittest
import numpy as np
import madpy.duration as dur
import madpy.tests.testdata.config as cfg


class TestAmplitude(unittest.TestCase):
    
    
    def test_measure_duration(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        df_dur = dur.measure_duration(st, cfg)
        self.assertEqual(len(df_dur), len(st))
        self.assertEqual(len(df_dur.columns), 8) 
        
        
    def test_coda_duration(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.

        duration, cc = dur.coda_duration(st[0], 10 ** -10, 'linear', cfg)
        self.assertAlmostEqual(duration, 175, 0)
        self.assertAlmostEqual(cc, -0.72, 2)
        
        
    def test_moving_average(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        avg_time, avg_data = dur.moving_average(st[0].copy(), cfg.Duration)
        self.assertEqual(np.sum(st[0].data - avg_data), 0)
        cfg.Duration.moving_average_window = 2
        avg_time, avg_data = dur.moving_average(st[0].copy(), cfg.Duration)
        self.assertEqual(len(st[0].data) - len(avg_data), 
                         2 * st[0].stats.sampling_rate - 1)
        
        
    def test_time_relative_p(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        time = np.arange(0, len(st[0].data)) * 0.01 - 40
        self.assertEqual(np.sum(dur.time_relative_p(st[0].copy()) - time), 0)
        
        
    def test_log_envelope(self):
        
        data = np.array([-10, 0, 1, 10, 100, 0, -10, 1, 100, 10], dtype=float)
        envelope = dur.log_envelope(data)
        self.assertEqual(len(envelope[np.isnan(envelope)]), 4)
        self.assertEqual(len(envelope[~np.isnan(envelope)]), 6)
        
        
    def test_coda_fitting_window(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        time, data_0 = dur.moving_average(st[0].copy(), cfg.Duration)
        data = dur.log_envelope(data_0)
        i0, i1 = dur.coda_fitting_window(st[0].copy(), cfg.Duration, time, data, 10 ** -10)
        self.assertEqual(i0, 5631)
        self.assertEqual(i1, 19292)
        
        
    def test_search_window(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        time,_ = dur.moving_average(st[0].copy(), cfg.Duration)
        begin = dur.search_window(st[0].copy(), time, cfg.Duration, 'begin')
        end = dur.search_window(st[0].copy(), time, cfg.Duration, 'end')
        self.assertEqual(begin, 4701)
        self.assertEqual(end, 6901)
    
        
    def test_search_window_seconds(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        begin = dur.search_window_seconds(st[0].copy(), cfg.Duration, 'begin')
        end = dur.search_window_seconds(st[0].copy(), cfg.Duration, 'end')
        self.assertEqual(begin, 8)
        self.assertEqual(end, 30)
        
        
    def test_phase_relative_p(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        self.assertEqual(dur.phase_relative_p(st[0], 'O'), -10)
        self.assertEqual(dur.phase_relative_p(st[0], 'P'), 0)
        self.assertEqual(dur.phase_relative_p(st[0], 'S'), 10)
        
        
    def test_fitting_window_start(self):
        
        self.assertEqual(dur.fitting_window_start(cfg.Duration, 20, 50), 20)
        cfg.Duration.start_fit_max = 3
        self.assertEqual(dur.fitting_window_start(cfg.Duration, 20, 50), 30)
        cfg.Duration.start_fit_max = 10
        self.assertEqual(dur.fitting_window_start(cfg.Duration, 20, 50), 23)
        
        
    def test_fitting_window_end(self):
        
        st = obspy.read('testdata/duration.mseed')
        st[0].stats.o = obspy.UTCDateTime('2020-10-10T13:05:00.00')
        st[0].stats.p = 10.
        st[0].stats.s = 20.
        
        _,data_0 = dur.moving_average(st[0].copy(), cfg.Duration)
        data = dur.log_envelope(data_0)
        end = dur.fitting_window_end(st[0], cfg.Duration, data, 5631, 10 ** -10)
        self.assertEqual(end, 19292)
        
        
    def test_coda_line_fit(self):
        
        m = -5
        b = 20
        x = np.arange(0, 1000, 0.1)
        y = m * x + b
        
        fit = dur.coda_line_fit(x, y)
        self.assertAlmostEqual(fit[1], m)
        self.assertAlmostEqual(fit[0], b)
        
        
    def test_get_duration_index(self):
        
        m = -5
        b = 20
        x = np.arange(0, 1000, 0.1)
        y = m * x + b
        
        i, cross = dur.get_duration_index(cfg.Duration, [b, m], 10 ** -10)
        self.assertAlmostEqual(i, 6.001)
        self.assertEqual(cross, 6001)
        
        
    def test_extend_line(self):
        
        x, y = dur.extend_line(-5, 20, 0, 500, 0.1)
        self.assertEqual(len(x), 500 / 0.1)
        self.assertEqual(len(y), 500 / 0.1)
         
        
    def test_coda_line_end(self):
        
        self.assertEqual(dur.coda_line_end(cfg.Duration, 10 ** -10), -10)
        cfg.Duration.threshold_type = 'noise'
        self.assertAlmostEqual(dur.coda_line_end(cfg.Duration, 10 ** -10), -20)
        cfg.Duration.threshold_type = 'absolute'
        
        
    def test_get_correlation_coefficient(self):
        
        line = np.array([-5, -1])
        time = np.arange(0, 1000, 0.01)
        data = line[1] * time + line[0]
        cc = dur.get_correlation_coefficient(line, time, data, 5, 90)
        self.assertAlmostEqual(cc, -1, 4)
        self.assertWarns(UserWarning, dur.get_correlation_coefficient, 
                         line, time, data, 5, 1110000)
               
        
    def test_calculate_cc(self):
        
        x = np.arange(0, 1000, 0.5)
        y = 5 * x - 1
        y_true = np.column_stack((x, y))
        self.assertRaises(AssertionError, dur.calculate_cc, y, y)
        self.assertEqual(dur.calculate_cc(y_true, y_true)[0, 3], 1)
        
        
    def format_output(self):
        
        data = ['2020', '13:00:00', 1., 2., 3.]
        self.assertRaises(AssertionError, dur.format_output, data)
        
    
if __name__ == '__main__':
    unittest.main()