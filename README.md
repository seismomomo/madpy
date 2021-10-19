Okay what the fuck are we going to do?

Each trace needs earthquake location (xyzt) and station location (xyz) 
coordinates: decimal degrees
depth/elevation: km

Each trace needs the origin time, P- and S- arrivals
origin time: UTCDateTime (obspy object)
arrivals: seconds since origin time
The trace data must include all arrivals

duration MUST BE time between p arrival and until end of noise window

if end_fit_threshold == absolute, then the fit goes until end_fit_wrt_noise, then where the line intersects with duration_absolute_threshold is the duration

if end_fit_threshold == noise, then the fit goes until end_fit_wrt_noise, then where the line intersects with duration_prep_noise is the duration

start_fit_wrt_max is an int and represents the fraction. so if 4, it means 1/4. if it equals 1, that means start at max amplitude. cannot equal 0.

PLOT_PHASE in plotting.py refers to what phase the time axis is relative to.

