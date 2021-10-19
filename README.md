# MADPy
### <ins>M</ins>easure <ins>A</ins>mplitude and <ins>D</ins>uration in <ins>Py</ins>thon 
#### Intended for seismic time series analysis

_____


<p>This is a Python-based tool that measures the amplitude and duration of a seismogram. The amplitudes and durations can be used in many seismic applications, such as magnitude calculation and seismic source discrimination.</p>

<p>MADPy relies heavily on <a href=https://github.com/obspy/obspy>Obspy</a>. The tool reads in Obspy Stream objects for measurement. Each Stream is comprised of Obspy Traces for each seismogram. The Trace must include the origin time, P- arrival, and S- arrival. Additionally, the Trace data must be pre-processed and ready for measurement. This tool does not include any post-processing.</p>

*Amplitude*
<p>The amplitude is defined as half the maximum peak-to-peak amplitude of the seismogram following <a href=https://doi.org/10.1785/0120060114>Pechmann et al. (2007)</a>. This is calculated by finding the maximum difference between inflection points. The signal to noise ratio is calculated for each amplitude measurement for quality control. </p>

*Duration*
<p>The duration is defined as the time from the P- arrival until the seismic energy reaches a user-specified energy threshold. A brief description of the method can be found in <a href=https://doi.org/10.1785/0120200188>Koper et al. (2021)</a>. It is calculated by fitting a line to the coda envelope in log space. The correlation coefficient between the line and the data is calculated for each duration measurement for quality control. </p>

The user can define measurement windows and other parameters in the [config](madpy/config.py) file.