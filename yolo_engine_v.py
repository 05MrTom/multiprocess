#!/usr/bin/env python
"""
trt_inference_async_v3.py

A standalone module to load a TensorRT TRT10 .engine file and run asynchronous inference
using execute_async_v3.
Usage:
    python trt_inference_async_v3.py path/to/model.engine --image path/to/image.jpg --input_shape 1 3 640 640
"""

import argparse
import json
import logging
from collections import namedtuple, OrderedDict

import numpy as np
import torch
import tensorrt as trt
import matplotlib.pyplot as plt
import cv2

# from PIL import Image
# import torchvision.transforms as transforms
import time
from contextlib import contextmanager
import contextlib


# # Configure logger
# LOGGER = logging.getLogger("TensorRTInference")
# LOGGER.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO)

# Configure logger
LOGGER = logging.getLogger("TensorRTInference")
LOGGER.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
LOGGER.addHandler(file_handler)
LOGGER.addHandler(console_handler)

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000  # in ms
    elapsed = (time.perf_counter() - start) * 1000
    LOGGER.info(f"{name} took {elapsed:.2f} ms")


# Dummy helper functions.
def check_requirements(package):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_version(actual_version, version_constraint, hard=False, msg=""):
    # Implement version checking if required
    pass

class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()



# Binding structure.
Binding = namedtuple("Binding", ("name", "idx", "dtype", "shape", "data", "ptr"))


class TensorRTInferenceEngine:
    def __init__(self, engine_path, device=None):
        """
        Initialize the TensorRT inference engine for TRT10.
        :param engine_path: Path to the .engine file.
        :param device: Torch device (defaults to CUDA if available).
        """
        self.engine_path = engine_path
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.context = None
        self.bindings = OrderedDict()  # Mapping from binding name to Binding tuple.
        self.output_names = []
        self.dynamic = False
        self.fp16 = False
        # For TRT10, is_trt10 is always True.
        self._load_engine()

    def _load_engine(self):
        """Load and deserialize the engine file and set up bindings (for TRT10)."""
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
            except Exception:
                LOGGER.info("No metadata found in engine file. Proceeding without metadata.")
                f.seek(0)
                metadata = {}

            dla = metadata.get("dla", None)
            if dla is not None:
                runtime.DLA_core = int(dla)

            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        try:
            self.context = self.engine.create_execution_context()
        except Exception as e:
            LOGGER.error(f"Error creating execution context: {e}")
            raise e

        # TRT10 branch only.
        num_tensors = self.engine.num_io_tensors
        for i in range(num_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            try:
                idx = self.engine.get_binding_index(name)
            except AttributeError:
                idx = i
            if is_input:
                shape = tuple(self.engine.get_tensor_shape(name))
                if -1 in shape:
                    self.dynamic = True
                    default_shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[1])
                    self.context.set_input_shape(name, default_shape)
                    shape = default_shape
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)
                shape = tuple(self.context.get_tensor_shape(name))
            tensor = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, idx, dtype, shape, tensor, int(tensor.data_ptr()))
        LOGGER.info("Engine loaded and bindings initialized.")

    def infer_async_v3(self, input_tensor):
        """
        Run asynchronous inference using execute_async_v3 (TRT10 only).
        """
        if self.device.type != "cuda":
            raise ValueError("Asynchronous inference with execute_async_v3 requires a CUDA device.")

        if not input_tensor.is_cuda:
            input_tensor = input_tensor.to(self.device)

        input_binding = self.bindings["images"]
        if self.dynamic and input_tensor.shape != input_binding.shape:
            self.context.set_input_shape("images", input_tensor.shape)
            self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
            for name in self.output_names:
                new_shape = tuple(self.context.get_tensor_shape(name))
                self.bindings[name].data.resize_(new_shape)
        expected_shape = self.bindings["images"].shape
        assert input_tensor.shape == expected_shape, (
            f"Input size {input_tensor.shape} does not match expected shape {expected_shape}"
        )

        for name, binding in self.bindings.items():
            if name == "images":
                self.context.set_tensor_address(name, int(input_tensor.data_ptr()))
            else:
                self.context.set_tensor_address(name, binding.ptr)

        stream = torch.cuda.Stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to the image shape.
    """
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, letterbox=True, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes from one image shape to another.
    """
    if letterbox:
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                   round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1))
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        if padding:
            boxes[..., 0] -= pad[0]
            boxes[..., 1] -= pad[1]
            if not xywh:
                boxes[..., 2] -= pad[0]
                boxes[..., 3] -= pad[1]
        boxes[..., :4] /= gain
        return clip_boxes(boxes, img0_shape)
    else:
        scale_x = img0_shape[1] / img1_shape[1]
        scale_y = img0_shape[0] / img1_shape[0]
        boxes[..., [0, 2]] *= scale_x
        boxes[..., [1, 3]] *= scale_y
        return clip_boxes(boxes, img0_shape)


def scale_points(img1_shape, points, img0_shape, letterbox=True, ratio_pad=None, normalize=False, padding=True):
    """
    Rescales coordinate points (x, y) from one image shape to another.
    """
    if letterbox:
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        points[..., 0] = (points[..., 0] - pad[0]) / gain
        points[..., 1] = (points[..., 1] - pad[1]) / gain
        points[..., 0] = points[..., 0].clamp(0, img0_shape[1])
        points[..., 1] = points[..., 1].clamp(0, img0_shape[0])
        if normalize:
            points[..., 0] /= img0_shape[1]
            points[..., 1] /= img0_shape[0]
        return points
    else:
        scale_x = img0_shape[1] / img1_shape[1]
        scale_y = img0_shape[0] / img1_shape[0]
        points[..., 0] *= scale_x
        points[..., 1] *= scale_y
        points[..., 0] = points[..., 0].clamp(0, img0_shape[1])
        points[..., 1] = points[..., 1].clamp(0, img0_shape[0])
        if normalize:
            points[..., 0] /= img0_shape[1]
            points[..., 1] /= img0_shape[0]
        return points


def xywh2xyxy(x):
    """
    Convert bounding box format from (x, y, w, h) to (x1, y1, x2, y2).
    """
    assert x.shape[-1] == 4, f"Expected last dim=4, got {x.shape}"
    y = x.new_empty(x.shape)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def pose_nms(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
    pre_nms_topk=1000
):
    """
    Optimized NMS for pose estimation.
    Args:
        prediction (torch.Tensor): Tensor with shape (batch, num_detections, detection_size)
                                   where detection_size >= 56 (first 4: x,y,w,h, 5th: conf, rest: keypoints).
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        max_det (int): Maximum detections per image.
        pre_nms_topk (int): Limit candidates to the top-K detections.
    Returns:
        List[torch.Tensor]: List of tensors (one per batch image) after NMS.
    """
    import torchvision

    # Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = []
    for pred in prediction:
        # Filter out detections below confidence threshold
        pred = pred[pred[:, 4] > conf_thres]
        if pred.shape[0] == 0:
            output.append(torch.empty((0, 56), device=pred.device))
            continue

        # Limit to top candidates to reduce NMS computation
        if pred.shape[0] > pre_nms_topk:
            scores = pred[:, 4]
            _, idx = scores.topk(pre_nms_topk)
            pred = pred[idx]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]  # Cap detections per image
        output.append(pred[keep])
    return output


if __name__ == "__main__":
    # For demonstration, engine and image paths are hardcoded.
    engine = TensorRTInferenceEngine("/home/amrit05/projects/shuttlengine/yolo11s-pose.engine")
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    profilers = (
                Profile(device=device),
                Profile(device=device),
                Profile(device=device),
            )
   # Placeholder for video path
    video_path = "/home/amrit05/.posedetection/2023-viktor-prannoy/rally_101047_101584.mp4"
    # Lists to store processing times
    frame_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        with timer("Frame Processing") as frame_time:
            with profilers[0]:
                # with timer("Preprocessing") as preprocess_time:
                    # orig_img_shape = frame.shape[:2]  # (height, width)
                    # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # input_tensor = transforms.Compose([
                    #     transforms.Resize((640, 640)),
                    #     transforms.ToTensor(),
                    # ])(image).unsqueeze(0).to(engine.device)
                with timer("Preprocessing") as preprocess_time:
                    orig_img_shape = frame.shape[:2]  # (height, width)
                    # Resize directly using OpenCV (optimized in C++)
                    resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                    # Convert from BGR to RGB
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    # Convert to a tensor: change shape from HxWxC to CxHxW and scale to [0,1]
                    input_tensor = (
                        torch.from_numpy(rgb_frame)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .div(255.0)
                        .to(engine.device)
                    )
                    
            # Run inference
            with profilers[1]:
                with timer("Inference") as inference_time:
                    output_tensor = engine.infer_async_v3(input_tensor)[0]

            # Post-processing
            with profilers[2]:
                with timer("Post-processing") as postprocess_time:
                    output_tensor = output_tensor.permute(0, 2, 1)
                    preds_nms = pose_nms(output_tensor, conf_thres=0.25, iou_thres=0.45, max_det=300, pre_nms_topk=1000)[0]
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

                #     for box, kpts in zip(boxes_np, keypoints_np):
                #         x1, y1, x2, y2 = box.astype(int)
                #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #         for (x, y, kp_conf) in kpts:
                #             if kp_conf > 0.25:
                #                 cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                # cv2.imshow("YOLO Pose Inference", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
        frame_times.append(frame_time())
        preprocess_times.append(preprocess_time())
        inference_times.append(inference_time())
        postprocess_times.append(postprocess_time())

    cap.release()
    cv2.destroyAllWindows()

    if frame_times:
        print(f"Avg. Preprocessing Time: {sum(preprocess_times) / len(preprocess_times):.2f} ms")
        print(f"Avg. Inference Time: {sum(inference_times) / len(inference_times):.2f} ms")
        print(f"Avg. Post-processing Time: {sum(postprocess_times) / len(postprocess_times):.2f} ms")
        print(f"Avg. Total Time: {sum(frame_times) / len(frame_times):.2f} ms")
        print(f"FPS: {1000 / (sum(frame_times) / len(frame_times)):.2f}")
