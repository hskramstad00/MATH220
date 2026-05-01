import numpy as np
import cv2
from math import pi
from djitellopy import Tello

# -----------------------------
# Distance & bearing functions
# -----------------------------
def calculate_dist(x_dist, fx, aruco_width):
    if x_dist == 0:
        return float('inf')
    return (aruco_width * fx) / x_dist


def calculate_bearing(center_x, cx, fx):
    dx = center_x - cx
    return -np.arctan(dx / fx)


# -----------------------------
# Connect to Tello
# -----------------------------
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_reader = tello.get_frame_read()

# -----------------------------
# Camera parameters (replace if calibrated)
# -----------------------------
fx = 960.33931
cx = 469.48031

aruco_width = 0.077  # meters

# -----------------------------
# ArUco setup
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# Main loop
# -----------------------------
while True:
    frame = frame_reader.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for corner, marker_id in zip(corners, ids):
            marker_id = marker_id[0]
            corner = corner[0]

            # Compute center
            center_x = int((corner[0][0] + corner[1][0]) / 2)
            center_y = int((corner[0][1] + corner[1][1]) / 2)

            # Width in pixels
            x_dist = abs(int(corner[1][0] - corner[0][0]))

            # Calculations
            dist = calculate_dist(x_dist, fx, aruco_width)
            bearing = calculate_bearing(center_x, cx, fx)

            # Drone height (optional)
            drone_height = tello.get_distance_tof() / 100  # meters

            # Draw marker
            cv2.aruco.drawDetectedMarkers(frame, [corner.reshape(1, 4, 2)])

            # Display info
            text = f"ID:{marker_id} D:{dist:.2f}m B:{bearing:.2f}rad H:{drone_height:.2f}m"
            cv2.putText(frame, text, (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            print(f"ID {marker_id}: Distance={dist:.2f}m, Bearing={bearing:.2f}rad, Height={drone_height:.2f}m")

    cv2.imshow("Tello ArUco", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

# -----------------------------
# Cleanup
# -----------------------------
tello.streamoff()
cv2.destroyAllWindows()