from dataclasses import dataclass

@dataclass(init=False)
class Amplitude:
    noise_phase: str = 'P' 
    noise_window_begin: float = -30.
    noise_window_end: float =  -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -1.
    signal_window_end: float = 30.
    plot: bool = False

@dataclass(init=False)
class Duration:
    noise_phase: str = 'O'
    noise_window_begin: float = -20.
    noise_window_end: float = -10.
    signal_phase: str = 'S'
    signal_window_begin: float = -2.
    signal_window_end: float = 20.
    moving_average_window: int = 0
    start_fit_max: int = 1
    end_fit_noise: float = 2.
    end_fit_threshold: str = 'absolute'
    duration_prep_noise: float = 2.
    duration_absolute_threshold: float = -10.
    plot: bool = False