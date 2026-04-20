import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class EKF3D:
    '''
    x: state [x, y, z]
    u_t: controlls [vx, vy, vz]
    Q: process noise
    R: meassurment noise


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

    def update_ARUCO(self, x, P, z_measured, marker_position):
        mx, my, mz = marker_position[0], marker_position[1], marker_position[2]
        dx = mx - x[0]
        dy = my - x[1]
        dz = mz - x[2]

        q2d = dx**2 + dy**2 
        q3d = q2d + dz**2
        r3d = np.sqrt(q3d)

        z_expected = np.array([
            # range (Aruco)
            r3d,
            # bearing (aruco)
            np.arctan2(dy, dx),
        ])

        H = np.array([
            [-dx / r3d, -dy / r3d,  -dz/r3d],
            [ dy / q2d, -dx / q2d,  0]
        ])
 
        # Innovation
        innovation = z_measured - z_expected
        innovation[1] = wrap_angle(innovation[1])


        # R from uncertantiy in Aruco meassurment
        R = self.R_aruco
 
        # Kalman gain and state update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
 
        x_new = x + K @ innovation
        P_new = (np.eye(3) - K @ H) @ P
 
        return x_new, P_new
    
    def update_TOF(self, x, P, z_measured):
        '''
        Update step for tof-sensor
        x: last position
        P: last covariance
        z_measured: the meassured height from tof-sensor
        '''

        
        # Expected meassurment
        z_expected = np.array([
            x[2]
        ])

        # H matrix would be identity since tof is linear
        H = np.array([
            [0, 0, 1]
        ])

        # innovation, only looking at 
        innovation = z_measured - z_expected

        # R from uncerntanity in TOF measurments
        R = self.R_tof

        # Kalman gain and state update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x_new = x + K @ innovation
        P_new = (np.eye(3) - K @ H) @ P

        return x_new, P_new




def calculate_dist(x_dist, fx, aruco_width):
    if x_dist == 0:
        return float('inf')
    return (aruco_width * fx) / x_dist


def calculate_bearing(center_x, cx, fx):
    dx = center_x - cx
    return -np.arctan(dx / fx)


def compute_speed(estimated_pose, waypoint):
    K_x = 30
    K_y = 50
    K_z = 30
    ex, ey, ez = estimated_pose
    wx, wy, wz = waypoint

    x1 = wx - ex
    y1 = wy- ey
    # rember drone_height is in cm, wz is in meters
    z1 = wz - ez

    x_cmd = x1 * K_x
    y_cmd = y1 * K_y
    z_cmd = z1 * K_z

    # clip speed
    max_speed = 15
    min_speed = -15
    max_speed_z = 15
 
    vy = int(np.clip(y_cmd,min_speed, max_speed)) 
    vx = int(np.clip(x_cmd, min_speed, max_speed))
    vz = int(np.clip(z_cmd, min_speed, max_speed_z))

    return vy, vx, vz

def do_something_speed(speed, min_value):
    if speed == 0:
        return 0
    if abs(speed) < min_value:
        return min_value if speed > 0 else -min_value
    return speed