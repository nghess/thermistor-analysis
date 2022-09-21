import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

size = 400
middle = int(size/2)
end = size-1

display = np.zeros((size, size, 3), dtype=np.uint8)
i = 0

k = 20
trail_raw = []
trail_avg = []

while True:
    i += 1  # Time
    therm = int(data[i]*size)-1  # Thermistor value

    #Clear frame
    display = np.zeros((size, size, 3), dtype=np.uint8)
    # Latest raw thermistor value
    display[therm, end] = (0, 0, 255)

    # Compile trail for raw values
    trail_raw.append(therm)
    if len(trail_raw) == end:
        trail_raw = trail_raw[1:]
    for t in range(1, len(trail_raw)):  # For loops are way too slow. Figure out alternative.
        display[trail_raw[-t], end-t] = (0, 0, 128)

    # Compile trail for AVG values
    if len(trail_raw) >= k:
        running_avg = int(np.mean(trail_raw[-k:]))
        trail_avg.append(running_avg)
        # Draw running avg trail
        for t in range(1, len(trail_avg)):
            display[trail_avg[-t], end-t] = (255, 0, 0)

    # Print timestamp
    cv2.putText(display, str(i), (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))

    # Clear out of frame values from trail
    if len(trail_raw) == end:
        trail_raw = trail_raw[1:]

    if len(trail_avg) == end:
        trail_avg = trail_avg[1:]



    cv2.imshow('Thermistor Signal', display)
    cv2.waitKey(16)

    if i == len(data):
        i = 0



