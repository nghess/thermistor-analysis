import numpy as np
import cv2

# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor1.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

# Display window dims
width = 1000
height = 500
middle = int(width/2)
end = width-1

# LSR Window
window = 20
db = 100  # debounce

i = 0  # initiate timestep
debounce = 0  # initiate debounce timer
trail_raw = []  # list to store data as it comes in
slope_compare = []  # compare current and previous slope to detect change
sniffs = []  # to store sniffs as they are detected


while True:
    i += 1  # Time
    debounce -= 1  # Debounce timer for sniffs

    therm = int((1-data[i])*height)-1  # Current thermistor value

    #Clear frame
    display = np.ones((height, width, 3), dtype=np.uint8)*64
    # Latest raw thermistor value
    display[therm, end] = (0, 0, 255)

    # Compile trail for raw values
    trail_raw.append(therm)
    if len(trail_raw) == end:
        trail_raw = trail_raw[1:]

    # Draw data
    for t in range(1, len(trail_raw)):  # For loops are slow. Figure out indexing alternative.
        display[trail_raw[-t], end-t] = (255, 255, 255)

    # Print timestamp
    cv2.putText(display, f'{i} ms', (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))

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
        if len(slope_compare) == 2 and slope_compare[1] > 1.1 and debounce <= 0:
            sniffs.append([width, int(y2)])
            debounce = db  # Reset debounce after detection

        # Draw circle at sniff detection
        for s in range(len(sniffs)):
            cv2.circle(display, sniffs[s], 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)

        # Decrement sniff marker x position
        for s in sniffs:
            s[0] -= 1

        # Draw LSR Line
        cv2.line(display, (width-window, int(y1)), (width, int(y2)), color, 1)

    # Display Results
    cv2.imshow('Thermistor Signal', display)
    cv2.imwrite("output/lsr-rt/" + str(i) + ".png", display)
    cv2.waitKey(8)

    # If all data has been read, restart feed
    if i == len(data)-1:
        i = 0