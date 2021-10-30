from dataclasses import dataclass

@dataclass(init=False)
class Amplitude:
    noise_phase: str = 'P' 
    noise_window_begin: float = -30.
    noise_window_end: float =  -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -1.
    signal_window_end: float = 30.
    amp_factor: float = 1.
    save_output: bool = True
    output_path: str = '.'
    plot: bool = True
    save_figure: bool = True
    figure_path: str = '.'

@dataclass(init=False)
class Duration:
    noise_phase: str = 'P'
    noise_window_begin: float = -30.
    noise_window_end: float = -1.
    signal_phase: str = 'S'
    signal_window_begin: float = -2.
    signal_window_end: float = 20.
    moving_average_window: int = 2
    start_fit_max: int = 1
    end_fit_noise: float = 2.
    threshold_type: str = 'absolute'
    duration_noise_threshold: float = 1.
    duration_absolute_threshold: float = -7.763462738511306
    save_output: bool = True
    output_path: str = '.'
    plot: bool = True
    save_figure: bool = True
    figure_path: str = '.'