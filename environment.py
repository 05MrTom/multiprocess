
from .common import ComputeProfile
from .pose import PoseDetector
from .shuttletrack import TinyTracker

class ShuttleEnvironment:
    def __init__(self, compute_profile: ComputeProfile, pose_detector: PoseDetector, 
                 shuttle_tracker: TinyTracker):
        self.compute_profile: ComputeProfile = compute_profile
        self.pose_detector: PoseDetector = pose_detector
        self.shuttle_tracker: TinyTracker = shuttle_tracker

    def __str__(self):
        return f"ShuttleEnvironment(name={self.compute_profile})"

    def __repr__(self):
        return self.__str__()
    
    def start(self):
        pass

    def stop(self):
        pass
