# Imports
from math import sin, cos, pi
import numpy as np
from djitellopy import Tello
import cv2
import cv2.aruco
from time import sleep
import time
from ekf import EKF3D, calculate_dist, calculate_bearing
import yaml

yaml_file = "example_map.yaml"
 
def read_marker_file():
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        markers = {
            item["marker_id"]: (
                item["position"][0],
                item["position"][1],
                item["position"][2]
            )
            for item in data
        }
    return markers
 
markers = read_marker_file()


# Imports
tello = Tello()
tello.connect()
tello.stream_on()

# give time to connect
sleep(2)
print(f'Battery status is: {tello.get_battery()}')

# setup for ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
aruco_width = 0.077  # in meters

# cameara parameters
fx = 960.33931
cx = 469.48031

fy = 960.67856
cy = 389.88603

# prosess støy
Q = np.diag([0.01, 0.01, 0.01])

# aruco støy. range bearing
R_aruco = np.diag([0.15**2, 0.1**2])

R_tof = np.array([[0.05**2]])

ekf = EKF3D(Q, R_aruco, R_tof)

x = np.array([0.0, 0.0, 1.0])
P = np.diag([0.5, 0.5, 0.1])
 
has_taken_off = False
frame_reader = tello.get_frame_read()
last_time = time.time()

while True:
    frame = frame_reader.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now = time.time()
    dt = now - last_time
    last_time = now
 
    # ---- Prediksjon (dronen hover, u ≈ 0) ----
    u = np.array([0.0, 0.0, 0.0])
    x, P = ekf.prediction(x, P, u, dt)
 
    # ---- Detekter ArUco ----
    corners, ids, _ = detector.detectMarkers(gray)
 
    tof_height = tello.get_distance_tof() / 100.0  # cm → m
 
    n_markers = 0
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
 
        for corner, marker_id in zip(corners, ids.flatten()):
            if marker_id not in markers:
                continue
 
 
            # Range
            x_dist  = int(corner[1][0] - corner[0][0])
            range_meas = calculate_dist(x_dist, fx, aruco_width)
 
            if range_meas == float('inf') or range_meas > 10.0:
                continue
 
            # Bearing
            center_x = int((corner[0][0] + corner[1][0]) / 2)
            bearing_meas = calculate_bearing(center_x, cx, fx)
 
            # z_measured: [range, bearing, height]
            z_measured = np.array([range_meas, bearing_meas, tof_height])
 
            # EKF update
            x, P = ekf.update(x, P, z_measured, markers[marker_id])
            n_markers += 1
 
            # Tekst per markør
            cv2.putText(frame,
                        f"ID{marker_id} r={range_meas:.2f}m b={np.degrees(bearing_meas):.1f} deg",
                        (int(center_x), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
    # ---- Vis EKF-estimat ----
    cv2.putText(frame, f"EKF: x={x[0]:.2f} y={x[1]:.2f} z={x[2]:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    cv2.putText(frame, f"ToF: {tof_height:.2f}m",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    cv2.putText(frame, f"P diag: [{P[0,0]:.4f}, {P[1,1]:.4f}, {P[2,2]:.4f}]",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
 
    cv2.putText(frame, f"Markers: {n_markers}  dt: {dt:.3f}s",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
 
    cv2.imshow("EKF Test", frame)
 
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord('e') and not has_taken_off:
        tello.takeoff()
        sleep(0.5)
        tello.move_up(50)
        has_taken_off = True
        last_time = time.time()
        print("Taken off!")
 
    if key == ord('q'):
        if has_taken_off:
            tello.send_rc_control(0, 0, 0, 0)
            sleep(0.3)
            tello.land()
        break
 
tello.streamoff()
cv2.destroyAllWindows()