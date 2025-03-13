import numpy as np

class Results:
    def __init__(self, boxes: np.ndarray, keypoints: np.ndarray):
        self.boxes = boxes  # Store boxes
        self.keypoints = keypoints  # Store keypoints

    def __repr__(self):
        return f"Results(boxes={self.boxes}, keypoints={self.keypoints})"