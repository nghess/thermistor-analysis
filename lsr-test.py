import numpy as np
import cv2
import matplotlib.pyplot as plt

# Display window dims
width = 500
height = 300
middle = int(width/2)
span = width-1

# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

window = 5
start = 110
stop = start + window

# LSR
A = np.vstack([time, np.ones(len(time))]).T
m, c = np.linalg.lstsq(A[start:stop], data[start:stop], rcond=None)[0]
print(m, c)

# Clear frame
display = np.zeros((height, width, 3), dtype=np.uint8)

for t in range(0, width-1):  # For loops are way too slow. Figure out alternative.
    display[int(data[t]*height), t] = (0, 0, 255)

cv2.line(display, (0, int(c*height)), (width, int(m*height)), (0, 255, 255), 2)

#cv2.imshow('Thermistor Signal', display)
#vc2.waitKey(100000000)


_ = plt.plot(time[:width], data[:width], 'o', label='Original data', markersize=1)
_ = plt.plot(time[start:stop], m*np.asarray(time[start:stop]) + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()

