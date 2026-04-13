import numpy as np
from bearing_and_range import calculate_bearing, calculate_dist

def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class EKF3D:
    '''
    x: state [x, y, z]
    u_t: controlls [vx, vy, vz]


    '''
    def __init__(self, Q, R_aruco, R_tof):

        self.Q = Q
        self.R_aruco= R_aruco
        self.R_tof = R_tof

    def prediction(self, x, P, u, dt):

        # state prediction
        x_pred = x + u * dt

        F = np.eye(3)

        # covariance prediction
        P_pred = F @ P @ F.T + self.Q

        return x_pred, P_pred

    def update(self, x, P, z_measured, marker_position):
        mx, my = marker_position[0], marker_position[1]
        dx = mx - x[0]
        dy = my - x[1]

        q = dx**2 + dy**2
        r = np.sqrt(q)

        z_expected = np.array([
            # range (Aruco)
            r,
            # bearing (aruco)
            np.arctan2(dx, dy),
            # tof-sensor height
            x[2]
        ])

        H = np.array([
            [-dx / r, -dy / r,  0],
            [ dy / q, -dx / q,  0],
            [ 0,       0,       1],
        ])
 
        # ---- Innovation ----
        innovation = z_measured - z_expected
        innovation[1] = wrap_angle(innovation[1])
 
        # ---- Kalman gain and state update ----
        S = H @ P @ H.T + self.R
        K = P @ H.T @ np.linalg.inv(S)
 
        x_new = x + K @ innovation
        P_new = (np.eye(3) - K @ H) @ P
 
        return x_new, P_new

