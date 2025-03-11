#!/usr/bin/env python
"""
trt_inference_async_v3.py

A standalone module to load a TensorRT .engine file and run asynchronous inference using execute_async_v3.
Usage:
    python trt_inference_async_v3.py path/to/model.engine --image path/to/image.jpg --input_shape 1 3 640 640 --async_v3
"""

import argparse
import json
import logging
from collections import namedtuple, OrderedDict

import numpy as np
import torch
import tensorrt as trt
import matplotlib.pyplot as plt



from PIL import Image
import torchvision.transforms as transforms

# Configure logger
LOGGER = logging.getLogger("TensorRTInference")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Dummy helper functions. Modify as needed.
def check_requirements(package):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_version(actual_version, version_constraint, hard=False, msg=""):
    # Implement version checking if required
    pass

# Environment flags; adjust these based on your setup.
IS_JETSON = False  # Set True if running on a Jetson board
LINUX = True       # Set True if running on Linux
PYTHON_VERSION = "3.8.10"  # Adjust as needed

# Updated binding structure to include a binding index.
Binding = namedtuple("Binding", ("name", "idx", "dtype", "shape", "data", "ptr"))

class TensorRTInferenceEngine:
    def __init__(self, engine_path, device=None):
        """
        Initialize the TensorRT inference engine.
        :param engine_path: Path to the .engine file.
        :param device: Torch device to run inference (defaults to CUDA if available).
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
        self.is_trt10 = None  # Will be determined during engine loading.
        self._load_engine()

    def _load_engine(self):
        """Load the engine file, deserialize it, and set up the bindings."""
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            # Try to read metadata (if present)
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
            except Exception:
                LOGGER.info("No metadata found in engine file. Proceeding without metadata.")
                f.seek(0)
                metadata = {}

            # Set DLA core if provided in metadata
            dla = metadata.get("dla", None)
            if dla is not None:
                runtime.DLA_core = int(dla)

            # Deserialize engine
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        # Create execution context
        try:
            self.context = self.engine.create_execution_context()
        except Exception as e:
            LOGGER.error(f"Error creating execution context: {e}")
            raise e

        # Determine if using TRT 10+ API.
        # If the engine does not have "num_bindings", assume TRT 10+ API.
        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        num_tensors = self.engine.num_io_tensors if self.is_trt10 else self.engine.num_bindings

        for i in range(num_tensors):
            if self.is_trt10:
                # For TRT 10+ API
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                try:
                    idx = self.engine.get_binding_index(name)
                except AttributeError:
                    idx = i
                if is_input:
                    shape = tuple(self.engine.get_tensor_shape(name))
                    # Check for dynamic shape (indicated by -1)
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
                # For pre-TRT 10 API
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                idx = i  # Use loop index since get_binding_index is not available.
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
            # Allocate buffer using a PyTorch tensor
            tensor = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, idx, dtype, shape, tensor, int(tensor.data_ptr()))
        LOGGER.info("Engine loaded and bindings initialized.")

    def infer(self, input_tensor):
        """
        Run synchronous inference on the input tensor.
        :param input_tensor: A torch.Tensor with shape matching the model's expected input.
        :return: List of output tensors.
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
        # For synchronous inference, update input tensor address.
        self.context.set_tensor_address("images", int(input_tensor.data_ptr()))
        self.context.execute_v2([b.ptr for b in self.bindings.values()])
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs

    def infer_async_v3(self, input_tensor):
        """
        Run asynchronous inference on the input tensor using execute_async_v3.
        This requires a TensorRT version that supports execute_async_v3.
        :param input_tensor: A torch.Tensor with shape matching the model's expected input.
        :return: List of output tensors.
        """
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

        # Update the tensor addresses using set_tensor_address and the stored tensor names.
        for name, binding in self.bindings.items():
            if name == "images":
                # For input, update the address from the provided tensor.
                self.context.set_tensor_address(name, int(input_tensor.data_ptr()))
            else:
                # For outputs, use the stored pointer.
                self.context.set_tensor_address(name, binding.ptr)

        stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        if stream is None:
            LOGGER.warning("No CUDA stream available. Falling back to synchronous inference.")
            return self.infer(input_tensor)

        if not hasattr(self.context, "execute_async_v3"):
            LOGGER.warning("execute_async_v3 not supported. Falling back to execute_async_v2.")
            return self.infer_async(input_tensor)

        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs

    def infer_async(self, input_tensor):
        """
        Run asynchronous inference on the input tensor using execute_async_v2.
        This is kept for comparison.
        :param input_tensor: A torch.Tensor with shape matching the model's expected input.
        :return: List of output tensors.
        """
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
        self.context.set_tensor_address("images", int(input_tensor.data_ptr()))
        stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        if stream is None:
            LOGGER.warning("No CUDA stream available. Falling back to synchronous inference.")
            return self.infer(input_tensor)
        self.context.execute_async_v2([b.ptr for b in self.bindings.values()], stream.cuda_stream)
        stream.synchronize()
        y = [self.bindings[name].data for name in sorted(self.output_names)]
        
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
    
    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): The bounding boxes to clip.
        shape (tuple): The shape of the image.

    Returns:
        (torch.Tensor | numpy.ndarray): The clipped boxes.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


import cv2

def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): Resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): The original image shape.
        ratio_pad (tuple): The ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y




def nms_rotated(boxes, scores, threshold=0.45, use_triu=True):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.
        use_triu (bool, optional): Whether to use `torch.triu` operator. It'd be useful for disable it
            when exporting obb models to some formats that do not support `torch.triu`.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    if use_triu:
        ious = ious.triu_(diagonal=1)
        # pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
        # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
        pick = torch.nonzero((ious >= threshold).sum(0) <= 0).squeeze_(-1)
    else:
        n = boxes.shape[0]
        row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
        col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
        upper_mask = row_idx < col_idx
        ious = ious * upper_mask
        # Zeroing these scores ensures the additional indices would not affect the final results
        scores[~((ious >= threshold).sum(0) <= 0)] = 0
        # NOTE: return indices with fixed length to avoid TFLite reshape error
        pick = torch.topk(scores, scores.shape[0]).indices
    return sorted_idx[pick]

import time


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.
        end2end (bool): If the model doesn't require NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin



def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords




def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords

if __name__ == "__main__":
   # Load TRT engine
    engine = TensorRTInferenceEngine("/home/amrit05/projects/shuttlengine/yolo11n-pose.engine")

    # Original image
    orig_image = Image.open("test.png").convert('RGB')
    orig_img_shape = orig_image.size[::-1]

    # Preprocess
    input_tensor = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])(orig_image).unsqueeze(0).to(engine.device)

    # Run inference
    output_tensor = engine.infer_async_v3(input_tensor)[0]
    print("Shape after inference:", output_tensor.shape)
    # output_tensor = output_tensor.permute(0, 2, 1)
    # print("Shape after transpose:", output_tensor.shape)    

    # YOLO-specific NMS
    preds_nms = non_max_suppression(output_tensor)[0]

    # Validate shape
    print("Shape after correct YOLO NMS:", preds_nms.shape)

    if preds_nms is not None and len(preds_nms):
        # Bounding boxes extraction
        boxes = preds_nms[:, :4]
        print(f"Bounding Boxes: {boxes[0]}")
        boxes = scale_boxes((640, 640), boxes, orig_img_shape)
        print("bounding boxes after scale:", boxes[0])
        print("Bounding boxes shape after correct YOLO NMS:", boxes.shape)

        # Keypoints extraction correctly (17 keypoints)
        keypoints = preds_nms[:, 6:].reshape(-1, 17, 3)
        keypoints[..., :2] = scale_coords((640, 640), keypoints[..., :2], orig_img_shape)
        print("Keypoints shape after correct YOLO NMS:", keypoints.shape)

        # Draw bounding boxes and keypoints on the image
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