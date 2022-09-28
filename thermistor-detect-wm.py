import numpy as np
import cv2


# Import .dat and zip raw and smoothed into DF
data = np.fromfile('.dat/thermistor2.dat', dtype=float)
# Normalize
data = np.subtract(data, np.min(data)) / (np.max(data)-np.min(data))
# Time axis
time = [t for t in range(len(data))]

# Display window dims
width = 500
height = 500
mid_w = int(width/2)
mid_h = int(height/2)
end = width-1
margin = 100

i = 0  # Time step
step = 1  # Step for slope calculation
k = 10  # K for rolling mean
trail_raw = []
trail_avg = []
running_avg = 0  # var to report avg Y value
weighted_mu = np.asarray([w**(1/k) for w in range(1, k)])
differences = []  # list to check ranges between consecutive values

print(weighted_mu)

for x in range(1, len(data)):
    dif = abs(data[x]-data[x-1])
    differences.append(dif)

min_delta = min(differences)
max_delta = max(differences)

# Function to pull out slope between timesteps
def get_slope(traj, step, scale, img):
    p1 = np.asarray([step, traj[-1]])
    p2 = np.asarray([0, traj[-1-step]])
    vec = np.subtract(p1, p2)
    norm = np.linalg.norm(vec)
    vec = (vec / norm) * scale
    vec = np.add(vec, p1)
    vec = (int(vec[0]+width-margin), int(vec[1]))
    return p1, vec


change_tally = []
breath = (0, 0, 255)

while True:
    i += 1  # Time
    therm = int((1-data[i])*height)-1  # Thermistor value

    #Clear frame
    display = np.zeros((height, width, 3), dtype=np.uint8)
    # Latest raw thermistor value
    display[therm, end-margin] = (0, 0, 255)

    # Compile trail for raw values
    trail_raw.append(therm)
    #if len(trail_raw) == end-margin:
    #    trail_raw = trail_raw[1:]
    for t in range(1, len(trail_raw)):  # For loops are way too slow. Figure out alternative.
        accel = (max_delta*height) / abs(trail_raw[-t]-trail_raw[-t-1])
        #print(abs(trail_raw[-t]-trail_raw[-t-1]))
        display[trail_raw[-t], end-t-margin] = (0, 255*accel, 128)

    # Compile trail for AVG values
    if len(trail_raw) >= k:
        running_avg = int(np.mean(trail_raw[-k:]))
        trail_avg.append(running_avg)
    #    if len(trail_avg) == end-margin:
    #        trail_avg = trail_avg[1:]
        # Draw running avg trail
        if len(trail_avg) >= 3:
            # Get differential
            cur_diff = np.ediff1d(trail_avg[-2:])[0]
            change_tally.append(cur_diff)
        if len(change_tally) > 3:
            change_tally = change_tally[1:]
            #print(change_tally)

        for t in range(1, len(trail_avg)):
            if sum(change_tally) > 10:
                breath = (255, 0, 0)
            else:
                breath = (0, 0, 255)
            display[trail_avg[-t], end-t-margin] = breath

        if len(trail_avg) > step:
            # Show slope
            p1, vec = get_slope(trail_avg, step, 30, display)
            cv2.line(display, (width-margin, p1[1]), vec, color=(255, 255, 0), thickness=1)

    # Print avg Y Value
    if len(trail_raw) >= k:
        cur_y = round(1-(running_avg/height), 2)
        cv2.putText(display, str(cur_y), (width-margin+50, trail_avg[-1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 0))

    # Print timestamp
    cv2.putText(display, f'{i} ms', (5, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.25, color=(255, 255, 255))

    # Clear out of frame values from trail
    if len(trail_raw) == end-margin:
       trail_raw = trail_raw[1:]

    if len(trail_avg) == end-margin:
        trail_avg = trail_avg[1:]


    cv2.imshow('Thermistor Signal', display)
    cv2.waitKey(16)

    if i == len(data):
        i = 0



