import numpy as np
import time

# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor1.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
ts = [t for t in range(len(data))]

# Display window dims
width = 1000
height = 500
end = 20

# LSR Window
window = 20
db = 100  # debounce

i = 0  # initiate timestep
debounce = 0  # initiate debounce timer
trail_raw = []  # list to store data as it comes in
sniffs = []  # to store sniffs as they are detected
x_axis = [x for x in range(window-1)]

begin = time.time()
while True:
    i += 1  # Time
    debounce -= 1  # Debounce timer for sniffs

    therm = int((1-data[i])*height)-1  # Current thermistor value

    # Compile trail for raw values
    trail_raw.append(therm)
    if len(trail_raw) == window:
        trail_raw = trail_raw[1:]
        A = np.vstack([x_axis, np.ones(len(x_axis))]).T
        m, c = np.linalg.lstsq(A, trail_raw, rcond=None)[0]

        # Try to pull out sniffs
        if m > 1.1 and debounce <= 0:
            sniffs.append(i)
            debounce = db  # Reset debounce after detection

    # If all data has been read, restart feed
    if i == len(data)-1:
        finish = time.time()
        break

print(finish - begin)
print(len(sniffs))
print(i)