# Imports
from math import sin, cos, pi
import numpy as np
from djitellopy import Tello
import cv2
import cv2.aruco
from time import sleep
import time
from ekf import EKF3D, calculate_dist, calculate_bearing, compute_speed, do_something_speed
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

# Drone setup
tello = Tello()
tello.connect()
tello.streamon()

sleep(2)
print(f'Battery status is: {tello.get_battery()}')

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
aruco_width = 0.077  # meters

# Camera parameters
fx = 960.33931
cx = 469.48031
fy = 960.67856
cy = 389.88603

# EKF setup
Q = np.diag([0.01, 0.01, 0.01])
R_aruco = np.diag([0.15**2, 0.1**2])
R_tof = np.array([[0.05**2]])

ekf = EKF3D(Q, R_aruco, R_tof)

x = np.array([0.0, 0.0, 0.8])
P = np.diag([0.5, 0.5, 0.1])

has_taken_off = False
frame_reader = tello.get_frame_read()
last_time = time.time()

vx, vy, vz = 0, 0, 0
time_start = None

min_speed_y = 12
min_speed_z = 10

waypoint = np.array([
    [0.1, 0, 1],
    [1.7, 0, 1.2]
])

current_waypoint = 0
number_waypoint = len(waypoint)
waypoint_tolerance = 0.05

tello.takeoff()
has_taken_off = True
time_start = time.time()
last_time = time.time()
print("Taken off!")

while True:
    frame = frame_reader.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now = time.time()
    dt = now - last_time
    last_time = now

    tof_height = tello.get_distance_tof() / 100.0

    # Hold dronen i lufta
    if has_taken_off:
        tello.send_rc_control(-vy, vx, vz, 0)

    # EKF prediction
    u = np.array([vx / 100.0, vy / 100.0, vz / 100.0])
    x, P = ekf.prediction(x, P, u, dt)

    # Aruco deteksjon
    corners, ids, _ = detector.detectMarkers(gray)

    n_markers = 0
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for corner, marker_id in zip(corners, ids.flatten()):
            if marker_id not in markers:
                continue

            pts = corner[0]

            # Range
            x_dist = int(pts[1][0] - pts[0][0])
            range_meas = calculate_dist(x_dist, fx, aruco_width)

            if range_meas == float('inf') or range_meas > 10.0:
                continue

            # Bearing
            center_x = int((pts[0][0] + pts[1][0]) / 2)
            bearing_meas = calculate_bearing(center_x, cx, fx)

            # z_measured: [range, bearing]
            z_measured_Aruco = np.array([range_meas, bearing_meas])

            # EKF update
            x, P = ekf.update_ARUCO(x, P, z_measured_Aruco, markers[marker_id])
            n_markers += 1

            # Tekst per markør
            cv2.putText(frame,
                        f"ID{marker_id} r={range_meas:.2f}m b={np.degrees(bearing_meas):.1f} deg",
                        (int(center_x), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # z_meassured: [height]
    z_measured_TOF = np.array([tof_height])

    # EKF update form TOF sensor after looking at all arucos in the frame
    x, P = ekf.update_TOF(x, P, z_measured_TOF)

    # Waypoint styring
    check = (now - time_start) if time_start else 0

    if current_waypoint < number_waypoint:
        vy, vx, vz = compute_speed(x, waypoint[current_waypoint])
        vy = do_something_speed(vy, min_speed_y)
        vx = do_something_speed(vx, min_speed_y)
        vz = do_something_speed(vz, min_speed_z)

        ex, ey, ez = x
        wx, wy, wz = waypoint[current_waypoint]
        dist = np.linalg.norm([wx - ex, wy - ey, wz - ez])

        if dist < waypoint_tolerance:
            print(f'Reached waypoint {current_waypoint}')
            current_waypoint += 1

    elif current_waypoint >= number_waypoint and has_taken_off:
        tello.send_rc_control(0, 0, 0, 0)
        sleep(0.3)
        tello.land()
        break

    # Info tekst
    cv2.putText(frame, f"EKF: x={x[0]:.2f} y={x[1]:.2f} z={x[2]:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"ToF: {tof_height:.2f}m",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"P diag: [{P[0,0]:.4f}, {P[1,1]:.4f}, {P[2,2]:.4f}]",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    cv2.putText(frame, f"Markers: {n_markers}  dt: {dt:.3f}s",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    if current_waypoint < number_waypoint:
        cv2.putText(frame, f"WP {current_waypoint}: {waypoint[current_waypoint][:3]}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    cv2.putText(frame, f"CMD: vx={vx} vy={-vy} vz={vz}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("EKF Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if has_taken_off:
            tello.send_rc_control(0, 0, 0, 0)
            sleep(0.3)
            tello.land()
        break

tello.streamoff()
cv2.destroyAllWindows()