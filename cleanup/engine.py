import json
import torch
import numpy as np
import tensorrt as trt
from collections import namedtuple, OrderedDict
from abc import ABC, abstractmethod

# Binding structure.
Binding = namedtuple("Binding", ("name", "idx", "dtype", "shape", "data", "ptr"))

class TensorRTInferenceEngine(ABC):
    @abstractmethod
    def _load_engine(self):
        pass
    @abstractmethod
    def inference(self, input_tensor: torch.Tensor):
        pass

class YoloInferenceEngine(TensorRTInferenceEngine):
    def __init__(self, engine_path, device=None, half=False):
        """
        Initialize the TensorRT inference engine for TRT10.
        :param engine_path: Path to the .engine file.
        :param device: Torch device (defaults to CUDA if available).
        """
        self.engine_path = engine_path
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.fp16 = half
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.context = None
        self.bindings = OrderedDict()  # Mapping from binding name to Binding tuple.
        self.output_names = []
        self.dynamic = False
        self._load_engine()

    def _load_engine(self):
        """Load and deserialize the engine file and set up bindings (for TRT10)."""
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
            except Exception:
                f.seek(0)
                metadata = {}

            dla = metadata.get("dla", None)
            if dla is not None:
                runtime.DLA_core = int(dla)

            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        self.context = self.engine.create_execution_context()

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

    def inference(self, input_tensor):
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
