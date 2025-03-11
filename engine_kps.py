#!/usr/bin/env python
"""
trt_inference_async_v3.py

A standalone module to load a TensorRT .engine file and run asynchronous inference using execute_async_v3.
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

from PIL import Image
import torchvision.transforms as transforms
import time
from contextlib import contextmanager


# Configure logger
LOGGER = logging.getLogger("TensorRTInference")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000  # Convert to milliseconds
    elapsed = (time.perf_counter() - start) * 1000  # Convert to milliseconds
    LOGGER.info(f"{name} took {elapsed:.2f} ms")

# Dummy helper functions.
def check_requirements(package):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_version(actual_version, version_constraint, hard=False, msg=""):
    # Implement version checking if required
    pass

# Binding structure.
Binding = namedtuple("Binding", ("name", "idx", "dtype", "shape", "data", "ptr"))

class TensorRTInferenceEngine:
    def __init__(self, engine_path, device=None):
        """
        Initialize the TensorRT inference engine.
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
        self.is_trt10 = None  
        self._load_engine()

    def _load_engine(self):
        """Load and deserialize the engine file and set up bindings."""
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

        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        num_tensors = self.engine.num_io_tensors if self.is_trt10 else self.engine.num_bindings

        for i in range(num_tensors):
            if self.is_trt10:
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
            else:
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                idx = i
                if is_input:
                    shape = tuple(self.engine.get_binding_shape(i))
                    if -1 in shape:
                        self.dynamic = True
                        default_shape = tuple(self.engine.get_profile_shape(0, i)[1])
                        self.context.set_binding_shape(i, default_shape)
                        shape = default_shape
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                    shape = tuple(self.context.get_tensor_shape(i))
            tensor = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, idx, dtype, shape, tensor, int(tensor.data_ptr()))
        LOGGER.info("Engine loaded and bindings initialized.")

    def infer(self, input_tensor):
        """
        Run synchronous inference on the input tensor.
        """
        input_binding = self.bindings["images"]
        if self.dynamic and input_tensor.shape != input_binding.shape:
            if self.is_trt10:
                self.context.set_input_shape("images", input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    new_shape = tuple(self.context.get_tensor_shape(name))
                    self.bindings[name].data.resize_(new_shape)
            else:
                idx = self.bindings["images"].idx
                self.context.set_binding_shape(idx, input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    idx = self.bindings[name].idx
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(idx)))
        expected_shape = self.bindings["images"].shape
        assert input_tensor.shape == expected_shape, (
            f"Input size {input_tensor.shape} does not match expected shape {expected_shape}"
        )
        self.context.set_tensor_address("images", int(input_tensor.data_ptr()))
        self.context.execute_v2([b.ptr for b in self.bindings.values()])
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs

    def infer_async_v3(self, input_tensor):
        """
        Run asynchronous inference using execute_async_v3.
        Only asynchronous inference with execute_async_v3 is supported.
        """
        if self.device.type != "cuda":
            raise ValueError("Asynchronous inference with execute_async_v3 requires a CUDA device.")

        if not input_tensor.is_cuda:
            input_tensor = input_tensor.to(self.device)

        input_binding = self.bindings["images"]
        if self.dynamic and input_tensor.shape != input_binding.shape:
            if self.is_trt10:
                self.context.set_input_shape("images", input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    new_shape = tuple(self.context.get_tensor_shape(name))
                    self.bindings[name].data.resize_(new_shape)
            else:
                idx = self.bindings["images"].idx
                self.context.set_binding_shape(idx, input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    idx = self.bindings[name].idx
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(idx)))
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
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, letterbox=True, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes from one image shape to another.
    If letterbox=True, it assumes aspect ratio was preserved with padding.
    If letterbox=False, independent scaling factors are used.
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
    If letterbox=True, assumes the original resize preserved aspect ratio with padding.
    If letterbox=False, applies independent scaling.
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

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    pose=False,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    end2end=False,
):
    """
    Perform NMS on detections.
    For pose estimation, each detection vector is assumed to be:
      [x, y, w, h, conf, keypoints...]
    where keypoints constitute 51 values (17 keypoints × 3).
    """
    import torchvision

    if pose:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        output = []
        for pred in prediction:
            pred = pred[pred[:, 4] > conf_thres]
            if not pred.shape[0]:
                output.append(torch.empty((0, 56), device=pred.device))
                continue
            boxes = pred[:, :4]
            scores = pred[:, 4]
            idx = torchvision.ops.nms(boxes, scores, iou_thres)
            output.append(pred[idx][:max_det])
        return output

    # Standard detection NMS branch.
    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if in_place:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    else:
        prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        box, cls, mask = x.split((4, nc, nm), 1)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
        n = x.shape[0]
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
    return output

if __name__ == "__main__":
    engine = TensorRTInferenceEngine("/home/amrit05/projects/shuttlengine/yolo11n-pose.engine")
    with timer("Warmup"):
        orig_image = Image.open("test.png").convert('RGB')
        orig_img_shape = orig_image.size[::-1]  # (height, width)

        input_tensor = transforms.Compose([
            transforms.Resize((640, 640)),  # Forced resize (warping)
            transforms.ToTensor(),
        ])(orig_image).unsqueeze(0).to(engine.device)

    with timer("Inference"):
        output_tensor = engine.infer_async_v3(input_tensor)[0]
    print("Shape after inference:", output_tensor.shape)
    
    output_tensor = output_tensor.permute(0, 2, 1)
    print("Shape after transpose:", output_tensor.shape)

    with timer("NMS"):
        preds_nms = non_max_suppression(output_tensor, pose=True)[0]
    print("Shape after correct YOLO NMS:", preds_nms.shape)
    
    if preds_nms is not None and len(preds_nms):
        with timer("Postprocess"):
            boxes = preds_nms[:, :4]
            print(f"Bounding Boxes: {boxes[0]}")
            boxes = scale_boxes((640, 640), boxes, orig_img_shape, letterbox=False)
            print("Bounding boxes after scale:", boxes[0])
            print("Bounding boxes shape after correct YOLO NMS:", boxes.shape)

            keypoints = preds_nms[:, 5:].reshape(-1, 17, 3)
            keypoints[..., :2] = scale_points((640, 640), keypoints[..., :2], orig_img_shape, letterbox=False)
            print("Keypoints shape after correct YOLO NMS:", keypoints.shape)

        vis_img = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
        for box, kpts in zip(boxes.cpu().numpy(), keypoints.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (x, y, kp_conf) in kpts:
                if kp_conf > 0.25:
                    cv2.circle(vis_img, (int(x), int(y)), 3, (0, 0, 255), -1)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLO Pose: Boxes and Keypoints")
        plt.show()
    else:
        print("No detections found after NMS.")
