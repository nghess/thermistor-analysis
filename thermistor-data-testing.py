import numpy as np
import plotly.express as px
import pandas as pd

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    #try:
    window_size = np.abs(int(window_size))
    order = np.abs(int(order))
    #except ValueError:
    #    raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


data = np.fromfile('.dat/thermistor2.dat', dtype=float)
time = [t for t in range(len(data))]
dat_hat = savitzky_golay(data, 501, 3) # window size 51, polynomial order 3
ts = list(zip(time, data, dat_hat))

df = pd.DataFrame(ts, columns=['time', 'data', 'dat_hat'])

print(df.head())
raw = px.line(df[:1000], x='time', y='data')
smooth = px.line(df, x='time', y='dat_hat')

smooth.show()

#https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way