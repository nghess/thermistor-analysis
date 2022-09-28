import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

# Display window dims
width = 800
height = 300
middle = int(width/2)
span = width-1


i = 0  # timestep
k = 20  # K for rolling mean
trail_raw = []
trail_avg = []
running_avg = 0  # var to report avg Y value


def clear_overflow(end, *args):
    for x in args:
        if len(args) == end:
            args = args[1:]

while True:
    i += 1  # Time
    therm = int((1-data[i])*height)-1  # Thermistor value

    #Clear frame
    display = np.zeros((height, width, 3), dtype=np.uint8)
    # Latest raw thermistor value
    display[therm, span] = (0, 0, 255)

    # Compile trail for raw values
    trail_raw.append(therm)
    if len(trail_raw) == span:
        trail_raw = trail_raw[1:]
    for t in range(1, len(trail_raw)):  # For loops are way too slow. Figure out alternative.
        display[trail_raw[-t], span-t] = (0, 0, 128)

    # Compile trail for AVG values
    if len(trail_raw) >= k:
        running_avg = int(np.mean(trail_raw[-k:]))
        trail_avg.append(running_avg)
        # Draw running avg trail
        for t in range(1, len(trail_avg)):
            display[trail_avg[-t], span-t] = (255, 255, 0)

    # Print timestamp
    cv2.putText(display, f'{i} ms', (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))
    # Print avg Y Value
    cur_y = round(1-(running_avg/height), 2)
    cv2.putText(display, str(cur_y), (width-50, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 0))


    # Clear out of frame values from trails
    clear_overflow(span, trail_raw, trail_avg)


    cv2.imshow('Thermistor Signal', display)
    cv2.waitKey(16)

    if i == len(data):
        i = 0



