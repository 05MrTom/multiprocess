#!/usr/bin/env python
"""
trt_inference_async.py

A standalone module to load a TensorRT .engine file and run asynchronous inference.
Usage:
    python trt_inference_async.py path/to/model.engine --image path/to/image.jpg --input_shape 1 3 640 640
"""

import argparse
import json
import logging
from collections import namedtuple, OrderedDict

import numpy as np
import torch
import tensorrt as trt

# Configure logger
LOGGER = logging.getLogger("TensorRTInference")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# Dummy helper functions. Modify as needed.
def check_requirements(package):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_version(actual_version, version_constraint, hard=False, msg=""):
    # Implement version checking as needed
    pass

# Environment flags; adjust these based on your setup.
IS_JETSON = False  # Set True if running on a Jetson board
LINUX = True       # Set True if running on Linux
PYTHON_VERSION = "3.8.10"  # Adjust as needed

# For binding information
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

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
        self.bindings = OrderedDict()
        self.binding_addrs = OrderedDict()
        self.output_names = []
        self.dynamic = False
        self.fp16 = False
        self.is_trt10 = None
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

        # Determine if using TensorRT 10 or earlier API
        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        num_tensors = self.engine.num_io_tensors if self.is_trt10 else self.engine.num_bindings

        for i in range(num_tensors):
            if self.is_trt10:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    shape = tuple(self.engine.get_tensor_shape(name))
                    # Check for dynamic shape (indicated by -1)
                    if -1 in shape:
                        self.dynamic = True
                        # Use default shape from profile shape 0
                        default_shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[1])
                        self.context.set_input_shape(name, default_shape)
                        shape = default_shape
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                    shape = tuple(self.context.get_tensor_shape(name))
            else:  # For TensorRT versions before 10.0
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
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
            # Allocate buffer using PyTorch tensor
            tensor = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, tensor, int(tensor.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
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
                idx = self.engine.get_binding_index("images")
                self.context.set_binding_shape(idx, input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    idx = self.engine.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(idx)))
        expected_shape = self.bindings["images"].shape
        assert input_tensor.shape == expected_shape, (
            f"Input size {input_tensor.shape} does not match expected shape {expected_shape}"
        )
        self.binding_addrs["images"] = int(input_tensor.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs

    def infer_async(self, input_tensor):
        """
        Run asynchronous inference on the input tensor using a CUDA stream.
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
                idx = self.engine.get_binding_index("images")
                self.context.set_binding_shape(idx, input_tensor.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=input_tensor.shape)
                for name in self.output_names:
                    idx = self.engine.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(idx)))

        expected_shape = self.bindings["images"].shape
        assert input_tensor.shape == expected_shape, (
            f"Input size {input_tensor.shape} does not match expected shape {expected_shape}"
        )
        self.binding_addrs["images"] = int(input_tensor.data_ptr())

        stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        if stream is None:
            LOGGER.warning("No CUDA stream available. Falling back to synchronous inference.")
            return self.infer(input_tensor)

        self.context.execute_async_v2(list(self.binding_addrs.values()), stream.cuda_stream)
        stream.synchronize()
        outputs = [self.bindings[name].data for name in sorted(self.output_names)]
        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Engine Asynchronous Inference")
    parser.add_argument("engine_file", type=str, help="Path to the TensorRT engine file (.engine)")
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[1, 3, 640, 640],
        help="Input tensor shape as space separated integers (default: 1 3 640 640)",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a local image file to run inference on"
    )
    parser.add_argument(
        "--async", dest="async_mode", action="store_true", help="Use asynchronous inference mode"
    )
    args = parser.parse_args()

    trt_engine = TensorRTInferenceEngine(args.engine_file)

    # If an image file is provided, load and preprocess the image; otherwise, use a dummy tensor.
    if args.image:
        from PIL import Image
        import torchvision.transforms as transforms

        # Load the image and ensure it is in RGB mode
        image = Image.open(args.image).convert("RGB")
        # Create a preprocessing pipeline; resize to expected dimensions and convert to tensor.
        preprocess = transforms.Compose([
            transforms.Resize((args.input_shape[2], args.input_shape[3])),
            transforms.ToTensor(),
            # Optionally add normalization if your model requires it:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(trt_engine.device)
        LOGGER.info(f"Loaded image from {args.image} with shape: {input_tensor.shape}")
    else:
        input_tensor = torch.randn(*args.input_shape).to(trt_engine.device)
        LOGGER.info(f"Using dummy input tensor with shape: {input_tensor.shape}")

    if args.async_mode:
        LOGGER.info("Using asynchronous inference (execute_async_v2).")
        outputs = trt_engine.infer_async(input_tensor)
    else:
        LOGGER.info("Using synchronous inference (execute_v2).")
        outputs = trt_engine.infer(input_tensor)

    for idx, out in enumerate(outputs):
        print(f"Output {idx} shape: {out.shape}")
