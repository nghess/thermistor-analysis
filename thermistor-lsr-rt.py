import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

# Display window dims
width = 300
height = 300
middle = int(width/2)
end = width-1

i = 0  # timestep
trail_raw = []

while True:
    i += 1  # Time
    therm = int((1-data[i])*height)-1  # Thermistor value

    #Clear frame
    display = np.zeros((height, width, 3), dtype=np.uint8)
    # Latest raw thermistor value
    display[therm, end] = (0, 0, 255)

    # Compile trail for raw values
    trail_raw.append(therm)
    if len(trail_raw) == end:
        trail_raw = trail_raw[1:]
    for t in range(1, len(trail_raw)):  # For loops are way too slow. Figure out alternative.
        display[trail_raw[-t], end-t] = (0, 0, 128)

    # Print timestamp
    cv2.putText(display, f'{i} ms', (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))

    # Print Slope Value (once m is calculated, report slope here)



    cv2.imshow('Thermistor Signal', display)
    cv2.waitKey(16)

    if i == len(data):
        i = 0