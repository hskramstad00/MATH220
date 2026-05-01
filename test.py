import cv2
import numpy as np
import time

# ---------------- CAMERA CALIBRATION ----------------
camera_matrix = np.array([[920, 0, 320],
                          [0, 920, 240],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_length = 0.077  # meters

# ---------------- EKF (STATE: x, y, z) ----------------
x_est = np.array([0.0, 0.0, 0.0])
P_est = np.diag([0.1, 0.1, 0.1])

Q = np.diag([0.02, 0.02, 0.02])   # process noise
R = np.diag([0.01, 0.01, 0.01])   # measurement noise


def ekf_predict(x, P, dt):
    # No motion model (static assumption)
    F = np.eye(3)

    x_pred = x
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred


def ekf_update(x, P, z):
    H = np.eye(3)

    y = z - x  # innovation

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_new = x + K @ y
    P_new = (np.eye(3) - K @ H) @ P

    return x_new, P_new


# ---------------- ARUCO SETUP ----------------
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Marker 3D corners (centered)
obj_points = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

prev_time = time.time()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    # Time update
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # EKF prediction
    x_est, P_est = ekf_predict(x_est, P_est, dt)

    if ids is not None:
        for i, corner in enumerate(corners):

            img_points = corner[0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            tvec = tvec.flatten()

            # ---------------- EKF UPDATE ----------------
            z = np.array([tvec[0], tvec[1], tvec[2]])
            x_est, P_est = ekf_update(x_est, P_est, z)

            # Draw marker + axis
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Raw measurement (for comparison)
            cv2.putText(frame, f"RAW x: {tvec[0]:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"RAW y: {tvec[1]:.2f}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"RAW z: {tvec[2]:.2f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ---------------- DISPLAY EKF RESULT ----------------
    cv2.putText(frame, f"EKF x: {x_est[0]:.2f} m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"EKF y: {x_est[1]:.2f} m", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"EKF z: {x_est[2]:.2f} m", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("EKF ArUco Tracking (x, y, z)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()