# MADPy
### <ins>M</ins>easure <ins>A</ins>mplitude and <ins>D</ins>uration in <ins>Py</ins>thon 
#### For seismic time series analysis
<br>

## Overview
_____

MADPy is a Python package that measures the amplitude and duration of a seismogram. The amplitudes and durations can be used in many seismic applications, such as magnitude calculation and seismic source discrimination.

MADPy relies heavily on <a href=https://github.com/obspy/obspy>Obspy</a>. The tool reads in Obspy Stream objects for measurement. Each Trace within the Stream must include the origin time, P- arrival, and S- arrival. Additionally, the Trace data must be pre-processed and ready for measurement. This tool does not include any post-processing. A brief tutorial for MADPy can be found [here](tutorial/madpy.ipynb).<br><br>

<div>
    <figure style=text-align:left>
        <img src=tutorial/amp-WY.YNR.01.HHE.png width=550><br>
        <figcaption>Example amplitude measurement</figcaption>
    </figure>
</div>
<br><br>
<div>
    <figure style=text-align:left>
        <img src=tutorial/dur-log-WY.YNR.01.HHZ.png width=550><br>
        <figcaption>Example duration measurement</figcaption>
    </figure>
</div>
<br>

## Getting started
_____

MADPy relies heavily on the editing of the [configuration file](madpy/config.py). Users who do not expect to edit the config file often should install the package using pip. Users who will edit the config file, or who are unsure, should use MADPy as a module in their working directory.

The current version of MADPy is: 0.1.0
<br><br>

#### <ins>Pip install</ins>

To install MADPy using pip, type the following command in the desired Python environment.

```pip install madpy-seis```

If using anaconda, the source code will likely be located in

```[conda path]/envs/[environment]/lib/python3.9/site-packages/madpy```

_Note: Users should exercise caution if editing modules other than config.py_
<br><br>

#### <ins>Modular use</ins>

To use MADPy as a collection of modules, navigate to your working directory.

```cd [path]/madpy```

Then type one of the following into the command line:

```pip download --no-deps madpy-seis```


```git clone https://github.com/seismomomo/madpy```
<br><br>

## Noise Measurement
_____

The noise level is the root mean square of the data and is calculated for both amplitude and duration measurements. The procedure is outlined below. Parameters from the [config file](madpy/config.py) are __bolded__ for clarity.
1. <ins>Trim noise window</ins> – The waveform is trimmed to the user-specified noise window. The __noise_window_begin__ and __noise_window_end__ parameters define the noise window, and are relative to the arrival specified in __noise_phase__. 

2. <ins>Measure noise</ins> – The noise level is the RMS of the data within the trimmed window. It is used differently for the different measurement types.
    - Amplitude measurements: The noise level can be used as a quality constraint for the amplitude measurement by calculating the signal-to-noise ratio.
    - Duration measurements: The noise level is used to determine the fitting window for the best fit line. It can be used as a threshold for the duration value (see Duration Measurement details below).
<br><br>

## Amplitude measurement
_____

The amplitude is defined as a user-specified factor of the maximum peak-to-peak amplitude of the seismogram. The procedure is outlined below. Parameters from the [config file](madpy/config.py) are __bolded__ for clarity. 
1. <ins>Trim signal window</ins> – The waveform is trimmed to the user-specified signal window. The __signal_window_begin__ and __signal_window_end__ parameters define the signal window, and are relative to the arrival specified in __signal_phase__. _Note: The amplitude is the maximum peak-to-peak amplitude within the signal window. This does not guarantee that it is the maximum peak-to-peak amplitude of the full waveform. Choose the signal window carefully._
   
2. <ins>Measure amplitude</ins> – The peak-to-peak amplitude is measured by using differentials to isolate the inflection points. The maximum difference between these inflection points is the peak-to-peak amplitude. The __amp_factor__ parameter controls which amplitude gets reported. For example, __amp_factor__=0.5 indicates that the reported value is half the maximum peak-to-peak amplitude.

3. <ins>Amplitude output</ins> – The amplitude information for the Stream is returned as a pandas Dataframe. Users have the option to save this output to file by setting __save_output__ to True. The file name is "amp-output.csv" and is saved in the path specified in __output_path__.
   

3. <ins>Plot</ins> – If __plot__ is set to True, the module will generate the waveform with the maximum amplitude marked. The time axis is relative to PLOT_PHASE, a global parameter set in [amp.py](madpy/plotting/amp.py). The plotting parameters are available in [params.py](madpy/plotting/params.py). The plot must be generated if the user wishes to save the figure (__save_figure__=True). The figure is saved in __figure_path__ and is named "amp-[trace id].png."
<br><br>

## Duration measurement
_____

The duration is defined as the time from the P- arrival until the seismic energy reaches a user-specified energy threshold. The procedure is outlined below. Parameters from the [config file](madpy/config.py) are __bolded__ for clarity.
    
1. <ins>Apply smoothing</ins> – A moving average is applied to the data before duration measurement. __moving_average_window__ defines the averaging in seconds. For example, __moving_average_window__=2 means a 2-second moving average will be applied to the data. This parameter should be set to 0 if smoothing is not desired.

2. <ins>Convert to envelope</ins> – The coda envelope is calculated by taking the log10 of the real part of the Hilbert transform. Bad values are set to NaN at this stage.

3. <ins>Determine fitting window</ins> – The duration is measured using a line that is fit to the envelope data within the fitting window. Conventionally, the fitting window starts at the maximum value of the envelope within __signal_window_begin__ and __signal_window_end__ (with respect to __signal_phase__). The fitting window ends at __end_fit_noise__, which is a factor of the noise. If __end_fit_noise__=2, then the fitting window ends at twice the noise level. For this conventional fitting scheme, __start_fit_max__=1. To start the fitting window elsewhere, relative to the envelope maximum, __start_fit_max__ describes the inverse of the location. For example, __start_fit_max__=4 starts the fitting window 25% of the way between the envelope maximum and __end_fit_noise__. This is useful for envelopes that are curved in log space.

4. <ins>Fit line to coda</ins> – The duration measurement necessitates fitting a straight line to the coda envelope. This is done using an L2 norm following Gm=d, where m includes the slope and intercept of the best fit line. The constraints force the slope to be negative since that is sensible for coda decay. There is no constraint on the intercept. _Note: The threadpoolctl library is invoked here to prevent Numpy parallelization in the background_.

5. <ins>Measure duration</ins> – The duration occurs where the best fit line from step 4 intercepts a pre-defined ground motion threshold. There are two options for defining this threshold. 
    - __threshold_type__='absolute': The line must cross a static threshold that is specified in __duration_absolute_threshold__. For example, if __duration_absolute_threshold__=-7.7, then the duration is the time between the P- arrival and where the best fit line intersects with -7.7. Be sure to specify this parameter in log space.
    - __threshold_type__='noise': The line must cross a factor of the noise level that is specfied in __duration_noise_threshold__. For example, if __duration_noise_threshold__=1, then the duration is the time between the P- arrival and where the best fit line intersects with the noise level.<br>
    
    _Note: Oftentimes the best fit line has to be extrapolated to reach the duration threshold. Sometimes, this intersection will occur beyond the waveform segment it is provided. The duration module will raise a Warning if this occurs_.
    
6. <ins>Calculate correlation coefficient</ins> – The Pearson correlation coefficient (CC) between the best fit line and the data is calculated to provide a measure of quality control. The resulting CC value should be negative, since the relationship between time and ground motion is inversely proportional.

7. <ins>Duration output</ins> – The duration information for the Stream is returned as a pandas Dataframe. Users have the option to save this output to file by setting __save_output__ to True. The file name is "dur-output.csv" and is saved in the path specified in __output_path__.

8. <ins>Plot</ins> – If __plot__ is set to True, the module will return a duration plot. There are two options for the plot, and both are specified when calling the duration module.
    - 'linear': This option plots the normal time series with the phases and duration marked.
    - 'log': This option plots the envelope of the waveform that is used for the duration measurement. This plot includes phases, moving average, best fit line, duration, and ground motion threshold. The best fit line will become dashed if it is extrapolated.<br>
    
    The time axis is relative to PLOT_PHASE, a global parameter set in [dur.py](madpy/plotting/dur.py). The plotting parameters can be changed in [params.py](madpy/plotting/params.py). The plot must be generated if the user wishes to save the figure (__save_figure__=True). The figure is saved in __figure_path__ and is named "dur-[plot type]-[trace id].png." 
    
    _Note: The 'log' option is best for debugging. The 'linear' option is best for a quick check_.