# common.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Dict

import numpy as np
from multiprocessing import shared_memory, Value, Lock

from cv2 import VideoCapture


class ComputeProfile(Enum):
    DesktopCPU = 1
    DesktopGPU = 2
    Jetson = 3
    NXP = 4
    OpenVino = 5


# Define an abstract interface for frame listeners.
class FrameListener(ABC):
    @abstractmethod
    def frame_added(self, frame_counter: int, frame: np.ndarray) -> None:
        """Called when a new frame is added to the queue."""
        pass


# Mixin class that provides listener registration and notification.
class FrameListenerMixin:
    def __init__(self, *args, **kwargs):
        self._listeners: List[FrameListener] = []
        super().__init__(*args, **kwargs)
    
    def register_listener(self, listener: FrameListener) -> None:
        """Register a new listener to be notified when a frame is added."""
        self._listeners.append(listener)
    
    def unregister_listener(self, listener: FrameListener) -> None:
        """Unregister a listener."""
        self._listeners.remove(listener)
    
    def notify_listeners(self, frame_counter: int, frame: np.ndarray) -> None:
        """Notify all registered listeners with the new frame."""
        for listener in self._listeners:
            listener.frame_added(frame_counter, frame)


# Original FrameStore remains available for in-process use.
class FrameStore(FrameListenerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.frame_counter: int = 0
        self._frames: Dict[int, np.ndarray] = {}
    
    def enqueue(self, frame: np.ndarray) -> None:
        """Append a new frame to the queue and notify listeners."""
        self.frame_counter += 1
        self._frames[self.frame_counter] = frame
        self.notify_listeners(self.frame_counter, frame)
    
    def __getitem__(self, index: int) -> np.ndarray:
        """Allow indexing to peek at frames without removing them."""
        return self._frames[index]
    
    def pop(self) -> np.ndarray:
        """
        Remove and return the last frame added (LIFO behavior).
        Raises IndexError if the queue is empty.
        """
        if not self._frames:
            raise IndexError("pop from empty frame queue")
        return self._frames.pop(self.frame_counter)
    
    def __len__(self) -> int:
        return len(self._frames)


# New shared memory based frame store.
class SharedFrameStore(FrameListenerMixin):
    def __init__(self, max_frames: int, frame_shape: tuple, frame_dtype: np.dtype):
        """
        Initialize a shared memory block to store up to max_frames frames.
        The memory is interpreted as a NumPy array of shape (max_frames, *frame_shape).
        """
        super().__init__()
        self.max_frames = max_frames
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype

        # Calculate total memory required.
        self.frame_size = int(np.prod(frame_shape)) * np.dtype(frame_dtype).itemsize
        total_size = self.frame_size * max_frames
        
        # Create shared memory.
        self.shm = shared_memory.SharedMemory(create=True, size=total_size)
        self.buffer = np.ndarray((max_frames, *frame_shape), dtype=frame_dtype, buffer=self.shm.buf)
        
        # Shared counters for ring buffer (using multiprocessing.Value).
        self.write_index = Value('i', 0)
        self.count = Value('i', 0)
        self.lock = Lock()

    def enqueue(self, frame: np.ndarray):
        """
        Write a new frame into the ring buffer.
        If max_frames is exceeded, the oldest frames are overwritten.
        Also notifies any registered listeners.
        """
        with self.lock:
            idx = self.write_index.value
            np.copyto(self.buffer[idx], frame)
            self.write_index.value = (idx + 1) % self.max_frames
            if self.count.value < self.max_frames:
                self.count.value += 1
            current_count = self.count.value
        # Notify listeners with the current frame count and the new frame.
        self.notify_listeners(current_count, frame)
    
    def get_latest_frame(self) -> np.ndarray:
        """
        Retrieve a copy of the most recently enqueued frame.
        """
        with self.lock:
            if self.count.value == 0:
                raise IndexError("No frames in store")
            idx = (self.write_index.value - 1) % self.max_frames
            frame = self.buffer[idx].copy()
        return frame
            
    def cleanup(self):
        """
        Clean up the shared memory.
        """
        self.shm.close()
        self.shm.unlink()
