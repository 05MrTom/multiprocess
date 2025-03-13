import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from abc import ABC, abstractmethod

from .engine import YoloInferenceEngine
from .profile import Profile
from .nms import non_max_suppression
from .results import Results

class BaseKeypointPredictor(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Results:
        pass

class YoloPredictor(BaseKeypointPredictor):
    def __init__(self, engine_path: str, device: torch.device = None, half: bool = False):
        self.engine = YoloInferenceEngine(engine_path, device=device, half=half)
        self.device = device
        self.profilers = (
                Profile(device=device),
                Profile(device=device),
                Profile(device=device),
            )
    
    def predict(self, frame):
        with self.profilers[0]:
            orig_img_shape = frame.shape[:2]  # (height, width)
            # Resize using OpenCV
            resized_frame = cv2.resize(frame, (640, 640))
            # Convert to tensor
            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.engine.device)
        with self.profilers[1]:
            output_tensor = self.engine.inference(input_tensor)[0]
        with self.profilers[2]:
            output_tensor = output_tensor.permute(0, 2, 1)
            preds_nms = non_max_suppression(output_tensor, conf_thres=0.25, iou_thres=0.45, max_det=300, pre_nms_topk=1000)[0]
            if preds_nms is not None and preds_nms.shape[0]:
                scale_x = orig_img_shape[1] / 640
                scale_y = orig_img_shape[0] / 640

                boxes = preds_nms[:, :4]
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                keypoints = preds_nms[:, 5:].reshape(-1, 17, 3)
                keypoints[..., 0] *= scale_x
                keypoints[..., 1] *= scale_y
                boxes_np = boxes.cpu().numpy()
                keypoints_np = keypoints.cpu().numpy()
                return Results(boxes_np, keypoints_np)
            return Results(None, None)
                