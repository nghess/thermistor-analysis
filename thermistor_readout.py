import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

size = 300
middle = int(size/2)

display = np.zeros((size, size, 3), dtype=np.uint8)
i = 0
trail_raw = []
trail_avg = []

while True:
    i += 1
    therm = int(data[i]*size)-1

    display = np.zeros((size, size, 3), dtype=np.uint8)
    display[therm, middle] = (0, 0, 255)

    trail_raw.append(therm)
    if len(trail_raw) == middle:
        trail_raw = trail_raw[1:]
    for t in range(1, len(trail_raw)):
        display[trail_raw[-t], middle-t] = (0, 0, 128)

    cv2.imshow('display', display)
    cv2.waitKey(16)

    if i == len(data):
        i = 0



