"""A Kalman filter for one box that moves at a constant speed.

State x (8 numbers):  [cx, cy, w, h, vx, vy, vw, vh]
    cx, cy = box centre, w, h = width and height, v... = change per frame.
Measurement z (4 numbers): [cx, cy, w, h] = the detector's box. The detector gives no speed.
"""
import numpy as np


def tlwh_to_cxcywh(box):
    left, top, w, h = box
    return np.array([left + w / 2, top + h / 2, w, h], dtype=float)


def cxcywh_to_tlwh(box):
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=float)


class KalmanBox:
    # F: "new = old + speed" for the 4 box numbers; the speeds stay the same
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    # H: take the 4 box numbers out of the 8 (the detector only sees the box)
    H = np.eye(4, 8)

    def __init__(self, box_tlwh, pos_noise=1.0, vel_noise=0.01, meas_noise=1.0):
        self.x = np.zeros(8)
        self.x[:4] = tlwh_to_cxcywh(box_tlwh)        # start at the first box, speed 0
        # P: how unsure we are. Position: a little. Speed: very (we have never seen it).
        self.P = np.diag([10.0] * 4 + [1000.0] * 4)
        self.Q = np.diag([pos_noise] * 4 + [vel_noise] * 4)   # people do not move perfectly straight
        self.R = np.eye(4) * meas_noise                        # the detector's box is a bit noisy

    def predict(self):
        """Move one frame forward. Returns the predicted box (tlwh)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q     # each frame without a box: less sure
        return self.box()

    def update(self, box_tlwh):
        """Mix the prediction with the detector's box."""
        z = tlwh_to_cxcywh(box_tlwh)
        y = z - self.H @ self.x                            # how far the detector is from our guess
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)           # gain: how much to trust the detector
        self.x = self.x + K @ y                            # also corrects the speed
        self.P = (np.eye(8) - K @ self.H) @ self.P         # after a box: more sure

    def box(self):
        return cxcywh_to_tlwh(self.x[:4])
