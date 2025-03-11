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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Engine Inference")
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--async", dest="async_mode", action="store_true", help="Use asynchronous inference (execute_async_v2)")
    group.add_argument("--async_v3", dest="async_v3_mode", action="store_true", help="Use asynchronous inference (execute_async_v3)")
    args = parser.parse_args()

    trt_engine = TensorRTInferenceEngine(args.engine_file)

    # If an image file is provided, load and preprocess the image; otherwise, use a dummy tensor.
    if args.image:
        from PIL import Image
        import torchvision.transforms as transforms

        image = Image.open(args.image).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((args.input_shape[2], args.input_shape[3])),
            transforms.ToTensor(),
            # Add normalization here if your model requires it:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(trt_engine.device)
        LOGGER.info(f"Loaded image from {args.image} with shape: {input_tensor.shape}")
    else:
        input_tensor = torch.randn(*args.input_shape).to(trt_engine.device)
        LOGGER.info(f"Using dummy input tensor with shape: {input_tensor.shape}")

    if args.async_v3_mode:
        LOGGER.info("Using asynchronous inference (execute_async_v3).")
        outputs = trt_engine.infer_async_v3(input_tensor)
    elif args.async_mode:
        LOGGER.info("Using asynchronous inference (execute_async_v2).")
        outputs = trt_engine.infer_async(input_tensor)
    else:
        LOGGER.info("Using synchronous inference (execute_v2).")
        outputs = trt_engine.infer(input_tensor)

    for idx, out in enumerate(outputs):
        
        print(f"Output {idx}: {out.shape}")
        # print(f"Output {idx} shape: {out.shape}")
