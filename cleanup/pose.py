import numpy as np
import cv2


class Pose:
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (6, 12), (5, 11), (11, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, kplines=[], fullPose=False):
        # Instead of "if not kplines", we check if kplines is None or empty.
        if kplines is None or (hasattr(kplines, 'size') and kplines.size == 0) or (not hasattr(kplines, 'size') and not kplines):
            return

        keypoints = []
        self.score = 0

        for kp, line in enumerate(kplines):
            # If the keypoint is a string, split it; otherwise assume it is already a numeric list/array.
            if isinstance(line, str):
                parts = line.split()
            else:
                parts = line  # e.g. a list or numpy array with [x, y, confidence]

            # When not in fullPose mode, the old code expected an index as well.
            # YOLO output doesn’t include an index so we use the enumeration.
            if not fullPose:
                if len(parts) == 4:
                    i, px, py, score = parts
                else:
                    # When index is missing
                    px, py, score = parts
                    i = kp
            else:
                # If fullPose is True, assume only [px, py, score] is provided
                px, py, score = parts[:3]
                i = kp

            keypoints.append((int(i), np.array([float(px), float(py)])))
            self.score += float(score)

        self.init_from_kp(keypoints)

    def init_from_kparray(self, kparray):
        kp = np.array(kparray).reshape((17, 2))
        keypoints = []
        for i in range(17):
            keypoints.append((i, kp[i]))
        self.init_from_kp(keypoints)

    def init_from_kp(self, keypoints):
        self.kp = np.empty((17, 2))
        self.kp[:] = np.NaN

        for i, p in keypoints:
            self.kp[i] = p

        self.bx = [np.nanmin(self.kp[:, 0]), np.nanmax(self.kp[:, 0])]
        self.by = [np.nanmin(self.kp[:, 1]), np.nanmax(self.kp[:, 1])]
        
    def draw_skeleton(self, img, colour=(0, 128, 0), thickness=5):
        cimg = img.copy()
        for line in self.skeleton:
            X, Y = self.kp[line[0]], self.kp[line[1]]
            if any(np.isnan(X)) or any(np.isnan(Y)):
                continue
            # We sometimes fill in NaNs with zeros
            if sum(X) == 0 or sum(Y) == 0:
                continue
            p0, p1 = tuple(X.astype(int)), tuple(Y.astype(int))
            # For the legs, colour them and the ankles separately
            if line == (13, 15) or line == (14, 16):
                cimg = cv2.line(cimg, p0, p1, (0, 128, 128), thickness)
                cimg = cv2.circle(cimg, p1, 3, (128, 128, 0), thickness=-1)
            else:
                cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg

    def get_base(self):
        # Returns the midpoint of the two ankle positions
        # Returning one of the two points if theres a NaN
        # or a zero
        left_nan = self.kp[15][0] != self.kp[15][0] or self.kp[15][0] == 0
        right_nan = self.kp[16][0] != self.kp[16][0] or self.kp[16][0] == 0
        if left_nan:
            return self.kp[16]
        elif right_nan:
            return self.kp[15]
        elif left_nan and right_nan:
            return self.get_centroid()
        return (self.kp[15] + self.kp[16]) / 2.

    def get_centroid(self):
        n = 0
        p = np.zeros((2,))
        for i in range(17):
            if any(np.isnan(self.kp[i])) or max(self.kp[i]) == 0:
                continue

            n += 1
            p += self.kp[i]
        return p / n

    def can_reach(self, p, epsx=1.5, epsy=1.5):
        # if within (1+/-eps) of the bounding box then we can reach it
        dx, dy = self.bx[1] - self.bx[0], self.by[1] - self.by[0]
        return self.bx[0] - epsx * dx < p[0] < self.bx[1] + epsx * dx and \
               self.by[0] - epsy * dy < p[1] < self.by[1] + epsy * dy
