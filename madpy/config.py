from dataclasses import dataclass

@dataclass(init=False)
class Amplitude:
    noise_phase: str = 'P' 
    noise_window_begin: float = -30.
    noise_window_end: float =  -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -1.
    signal_window_end: float = 30.
    amp_plot: str = 'yes'

@dataclass(init=False)
class Duration:
    noise_phase: str = 'P'
    noise_window_begin: float = -30.
    noise_window_end: float = -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -2.
    signal_window_end: float = 20.
    moving_average_window: int = 2
    start_fit_wrt_max: int = 4
    end_fit_wrt_noise: float = 2.
    end_fit_threshold: str = 'absolute'
    duration_prep_noise: float = 1.
    duration_absolute_threshold: float = -7.763462738511306
    dur_plot: str = 'yes'