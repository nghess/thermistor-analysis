import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as spysig


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    import numpy as np
    from math import factorial

    window_size = np.abs(int(window_size))
    order = np.abs(int(order))

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
#normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))

time = [t for t in range(len(data))]
dat_hat = savitzky_golay(data, 101, 3) # window size 51, polynomial order 3
ts = list(zip(time, data, dat_hat))
df = pd.DataFrame(ts, columns=['time', 'data', 'dat_hat'])



#Smoothing data in python:
rolling_window = 21
therm_3 = np.convolve(dat_hat, np.ones((rolling_window,))/rolling_window, mode='same')
#rolling window - how much you want to smooth by - depends on each signal
#every 21 samples for Reese's data - mice sniff at 16 hz, don't want window bigger than 50 - could smooth out fast sniffs

peakwindow=21
localmaxima = spysig.argrelmax(therm_3, order=peakwindow)
localmaxima = localmaxima[0]

#for i in range(0,len(localmaxima)): #0 thru length of peaks (looping through all peaks)
#    plt.axvline(localmaxima[i]) #plot vertical line at index of sniff file that corresponds with the current peak
#plt.show()

print(localmaxima)

peaks = []

for i in localmaxima:
    peaks.append(data[i])

print(peaks)

peaks_zip = list(zip(localmaxima, peaks))
peaks_df = pd.DataFrame(peaks_zip, columns=['localmaxima', 'peaks'])

fraction = 2

#Plotly
fig_lines = px.line(df[:len(df)//fraction], x='time', y='dat_hat')
fig_point = px.scatter(peaks_df[:len(peaks_df)//fraction], x='localmaxima', y='peaks', color_discrete_sequence=['red'])

fig = go.Figure(data=fig_point.data+fig_lines.data)

fig.show()
