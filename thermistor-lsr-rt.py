import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

# Display window dims
width = 1000
height = 500
middle = int(width/2)
end = width-1

#LSR Window
window = 20

i = 0  # timestep
debounce = 0  # debounce timer
trail_raw = []  # list to store data as it comes in
slope_compare = []  # compare current and previous slope to detect change
sniffs = [] # to store sniffs as they are detected


while True:
    i += 1  # Time
    debounce -= 1  # debounce timer for sniffs

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
    #cv2.putText(display, f'{i} ms', (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))

    # LSR
    if len(trail_raw) >= window:
        x_axis = [x for x in range(width-window, width)]
        A = np.vstack([x_axis, np.ones(len(x_axis))]).T
        m, c = np.linalg.lstsq(A, trail_raw[-window:], rcond=None)[0]

        # Get points for line
        y1 = m*(width-window) + c
        y2 = m*width + c

        slope_compare.append(m)
        if len(slope_compare) == 3:
            slope_compare = slope_compare[1:]

        # Detect sniff via slope
        if m > 1:
            color = (255, 255, 0)
        else:
            color = (0, 0, 255)

        # Try to pull out sniffs
        if len(slope_compare) == 2 and slope_compare[1] > 1 > slope_compare[0] and debounce <= 0:
            sniffs.append([width, int(y2)])
            debounce = 75  # initiate debounce after detection

        for s in range(len(sniffs)):
            cv2.circle(display, sniffs[s], 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        for x in sniffs:
            x[0] -= 1

        # LSR Line
        cv2.line(display, (width-window, int(y1)), (width, int(y2)), color, 1)

    # Print Slope Value (once m is calculated, report slope here)


    cv2.imshow('Thermistor Signal', display)
    cv2.waitKey(8)

    if i == len(data)-1:
        i = 0