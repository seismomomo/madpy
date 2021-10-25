from dataclasses import dataclass

@dataclass(init=False)
class Amplitude:
    noise_phase: str = 'P' 
    noise_window_begin: float = -30.
    noise_window_end: float =  -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -1.
    signal_window_end: float = 30.
    plot: bool = True

@dataclass(init=False)
class Duration:
    noise_phase: str = 'P'
    noise_window_begin: float = -30.
    noise_window_end: float = -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -2.
    signal_window_end: float = 20.
    moving_average_window: int = 2
    start_fit_max: int = 4
    end_fit_noise: float = 2.
    threshold_type: str = 'absolute'
    duration_noise_threshold: float = 1.
    duration_absolute_threshold: float = -7.763462738511306
    plot: bool = True