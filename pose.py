from abc import ABC, abstractmethod
from typing import Tuple, List

from collections import defaultdict
import cv2

import numpy as np
import pandas as pd

from ultralytics import YOLO

# Removed: from .common import ComputeProfile, FrameListener, FrameStore
from court import Court

import time

class PoseDetector(ABC):
    @abstractmethod
    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect human pose keypoints in the provided frame.
        
        Args:
            frame (np.ndarray): An image frame.
        
        Returns:
            np.ndarray: A matrix with shape (n, 2), where n is the number of detected keypoints.
        """
        pass

    @abstractmethod
    def _postprocess(self, keypoints: np.ndarray) -> None:
        """
        Postprocess the detected keypoints to convert them into a more usable format.
        
        Args:
            keypoints (np.ndarray): A matrix with shape (n, 2), where n is the number of detected keypoints.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, one for the Top player and one for the Bottom player.
        """
        pass

    def frame_added(self, frame_counter: int, frame: np.ndarray) -> None:
        """Called when a new frame is added."""
        pose_vector = self._detect(frame)
        self._postprocess(frame_counter, pose_vector)
        

    def initialize_empty_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Initializes empty DataFrames for the bottom and top players.
        
        Returns:
            df_bottom (pd.DataFrame): Empty DataFrame for the bottom player.
            df_top (pd.DataFrame): Empty DataFrame for the top player.
        """
        self.columns = ["frame"] + [f"{joint}_{axis}" for joint in Pose.joint_names for axis in ["x", "y"]]
        self.df_bottom = pd.DataFrame(columns=self.columns)
        self.df_top = pd.DataFrame(columns=self.columns)
    

    def append_pose_to_dataframe(self, frame_id, poses):
        """
        Optimized function to append pose data for bottom and top players for a given frame.
        """
        num_joints = len(Pose.joint_names)
        bottom_kp = getattr(poses.get(1), 'kp', np.full((num_joints, 2), np.nan))
        top_kp = getattr(poses.get(2), 'kp', np.full((num_joints, 2), np.nan))
        row_bottom = np.hstack(([frame_id], bottom_kp.flatten()))
        row_top = np.hstack(([frame_id], top_kp.flatten()))
        self.df_bottom.loc[len(self.df_bottom)] = row_bottom
        self.df_top.loc[len(self.df_top)] = row_top

class YoloPoseDetector(PoseDetector):
    def __init__(self, court_pts: List[List[float]]) -> None:
        """
        Initializes the YOLO Pose Detector.
        
        Args:
            court_pts (List[List[float]]): List of court points.
        """
        self.model_path = "yolo11n-pose.pt"
        self.model = self._load_model(self.model_path)
        self.initialize_empty_dataframe()
        corners = [court_pts[2], court_pts[3], court_pts[1], court_pts[4]]
        self.court = Court(corners)
        
    def _load_model(self, model_path: str):
        """
        Dummy function to simulate loading a YOLO pose model.
        Replace this method with actual model loading logic.
        """
        print(f"Loading YOLO pose model from: {model_path}")
        model = YOLO(model_path)
        return model
    
    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect human pose keypoints in the provided frame.
        
        Args:
            frame (np.ndarray): An image frame.
        
        Returns:
            np.ndarray: A matrix with shape (n, 2), where n is the number of detected keypoints.
        """
        results = self.model(frame, verbose=False)#, imgsz=512, half=True)
        for result in results:
            keypoints = result.keypoints.data.cpu().numpy()
        return keypoints
    
    def _postprocess(self, frame_counter: int, keypoints: np.ndarray) -> None:
        filtered_poses = process_pose_data(keypoints, self.court, frame_counter, fullPose=False)
        for frame_id, poses in filtered_poses.items():
            self.append_pose_to_dataframe(frame_id, poses)

class Pose:
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (6, 12), (5, 11), (11, 12),       # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, kplines=[], fullPose=False):
        if kplines is None or (hasattr(kplines, 'size') and kplines.size == 0) or (not hasattr(kplines, 'size') and not kplines):
            return

        keypoints = []
        self.score = 0

        for kp, line in enumerate(kplines):
            if isinstance(line, str):
                parts = line.split()
            else:
                parts = line

            if not fullPose:
                if len(parts) == 4:
                    i, px, py, score = parts
                else:
                    px, py, score = parts
                    i = kp
            else:
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
            if sum(X) == 0 or sum(Y) == 0:
                continue
            p0, p1 = tuple(X.astype(int)), tuple(Y.astype(int))
            if line == (13, 15) or line == (14, 16):
                cimg = cv2.line(cimg, p0, p1, (0, 128, 128), thickness)
                cimg = cv2.circle(cimg, p1, 3, (128, 128, 0), thickness=-1)
            else:
                cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg

    def get_base(self):
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
        dx, dy = self.bx[1] - self.bx[0], self.by[1] - self.by[0]
        return self.bx[0] - epsx * dx < p[0] < self.bx[1] + epsx * dx and \
               self.by[0] - epsy * dy < p[1] < self.by[1] + epsy * dy

def process_pose_data(pose_data, court, frame_id, fullPose=False):
    if isinstance(pose_data, np.ndarray) and pose_data.ndim == 3:
        pose_data = [pose_data]

    filtered_poses = defaultdict(lambda: {1: None, 2: None})

    def filter_pose(fid, kp):
        if kp is None or len(kp) == 0:
            return
        pose = Pose(kp, fullPose=fullPose)
        base = pose.get_base()
        in_court = court.in_court(base, slack=[0.05, 0.15])
        if in_court in [1, 2]:
            if (filtered_poses[fid][in_court] is None or 
                pose.score > filtered_poses[fid][in_court].score):
                filtered_poses[fid][in_court] = pose

    for _, frame_poses in enumerate(pose_data):
        for kp in frame_poses:
            filter_pose(frame_id, kp)

    for _ in filtered_poses:
        for player in [1, 2]:
            if filtered_poses[frame_id][player] is None:
                filtered_poses[frame_id][player] = Pose()

    return filtered_poses
