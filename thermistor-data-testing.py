import numpy as np
import plotly.express as px
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
time = [t for t in range(len(data))]
dat_hat = savitzky_golay(data, 33, 3) # window size 51, polynomial order 3
ts = list(zip(time, data, dat_hat))
df = pd.DataFrame(ts, columns=['time', 'data', 'dat_hat'])

#Plotly
fig = px.line(df[:1000], x='time', y=df.columns)
fig.show()


#Smoothing data in python:
rolling_window = 21
therm_3 = np.convolve(dat_hat, np.ones((rolling_window,))/rolling_window, mode='same')
#rolling window - how much you want to smooth by - depends on each signal
#every 21 samples for Reese's data - mice sniff at 16 hz, don't want window bigger than 50 - could smooth out fast sniffs
#needs to be an odd number to center

plt.plot(therm_3)
#plt.xlim(1600,2400) #800 sample chunk is 1 second
#plt.show()

#Finding data peaks in python:

peakwindow=21
localmaxima = spysig.argrelmax(therm_3, order=peakwindow)
localmaxima = localmaxima[0]

#for i in range(0,len(localmaxima)): #0 thru length of peaks (looping through all peaks)
#    plt.axvline(localmaxima[i]) #plot vertical line at index of sniff file that corresponds with the current peak
#plt.show()

print(localmaxima)