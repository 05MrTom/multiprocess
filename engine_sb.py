import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from contextlib import contextmanager
import logging

from court import read_court
from pose import YoloPoseDetector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    elapsed = time.perf_counter() - start
    logging.info(f"{name} took {elapsed:.4f} seconds")

def worker_a(detector, shm_name, buffer_shape, frame_dtype,
             new_frame_event, frame_processed_event,
             termination_event, results_dict, current_index, buffer_size):
    """
    Worker A:
      - Waits for the new_frame_event.
      - Reads the latest frame from the circular buffer.
      - Runs pose inference (via detector.frame_added).
      - Extracts keypoints for top and bottom players.
      - Stores the results in the Manager dictionary keyed by frame id.
      - Signals with frame_processed_event so the server can proceed.
    """
    # Attach to the shared memory block holding the circular buffer.
    shm = shared_memory.SharedMemory(name=shm_name)
    circular_buffer = np.ndarray(buffer_shape, dtype=frame_dtype, buffer=shm.buf)
    
    processed_frame = 0
    while not termination_event.is_set():
        # Wait until a new frame is available.
        new_frame_event.wait()
        new_frame_event.clear()
        
        # Determine the index of the most recently written frame.
        with current_index.get_lock():
            # If current_index==0 then the latest frame is at index buffer_size - 1.
            idx = (current_index.value - 1) % buffer_size
        
        # Copy the frame from the circular buffer.
        frame = circular_buffer[idx].copy()
        # logging.info(f"Worker: Processing frame from buffer index {idx}")
        
        with timer("detector.frame_added"):
            detector.frame_added(processed_frame, frame)
        
        # Extract keypoints from detector's DataFrames.
        if not detector.df_top.empty and not detector.df_bottom.empty:
            top_keypoints = detector.df_top.iloc[-1].tolist()
            bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top_keypoints = []
            bottom_keypoints = []
        
        # Store results in the Manager dictionary.
        results_dict[processed_frame] = {"top": top_keypoints, "bottom": bottom_keypoints}
        # logging.info(f"Worker: Stored results for frame {processed_frame}")
        
        processed_frame += 1
        
        # Signal that processing of the current frame is complete.
        frame_processed_event.set()
    
    shm.close()

def server():
    """
    Server:
      - Opens the video and creates a shared memory circular buffer to hold 30 frames.
      - For each frame read from the video, writes it into the next slot of the circular buffer.
      - Signals Worker A to process the frame.
      - Waits for Worker A to finish processing before writing the next frame.
    """
    # Initialize the YOLO pose detector.
    court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
    detector = YoloPoseDetector(court_pts)
    
    video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"  # Replace with your video path.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read the first frame from video.")
        return
    frame_shape = frame.shape       # e.g., (height, width, channels)
    frame_dtype = frame.dtype
    
    # Circular buffer parameters
    buffer_size = 300
    buffer_shape = (buffer_size, *frame_shape)
    total_bytes = np.prod(buffer_shape) * frame_dtype.itemsize
    
    # Create shared memory for the circular buffer.
    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    circular_buffer = np.ndarray(buffer_shape, dtype=frame_dtype, buffer=shm.buf)
    
    # Shared index to track where the next frame should be written.
    current_index = mp.Value('i', 0)
    
    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Create Events for synchronization.
    new_frame_event = mp.Event()         # Server sets this after writing a frame.
    frame_processed_event = mp.Event()   # Worker A sets this after processing a frame.
    termination_event = mp.Event()       # Signal termination.
    
    # Initially, signal that the previous frame has been processed.
    frame_processed_event.set()
    
    workerA = mp.Process(target=worker_a, args=(
        detector, shm.name, buffer_shape, frame_dtype,
        new_frame_event, frame_processed_event,
        termination_event, results_dict,
        current_index, buffer_size
    ), name="Worker-A")
    workerA.start()
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps
    frame_counter = 0
    try:
        with timer("server loop"):
            while True:
                start_time = time.perf_counter()
                
                # Wait for Worker A to signal that it has processed the previous frame.
                frame_processed_event.wait()
                frame_processed_event.clear()
                
                ret, frame = cap.read()
                if not ret:
                    # logging.warning("End of video reached.")
                    break
                
                # Write the new frame into the circular buffer at the current index.
                with current_index.get_lock():
                    idx = current_index.value
                    np.copyto(circular_buffer[idx], frame)
                    current_index.value = (idx + 1) % buffer_size
                # logging.info(f"Server: Wrote frame {frame_counter} to buffer index {idx}")
                frame_counter += 1
                
                # Signal Worker A that a new frame is ready.
                new_frame_event.set()
                
                elapsed_time = time.perf_counter() - start_time
                time_to_wait = frame_interval - elapsed_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt, shutting down.")
    finally:
        with timer("server shutdown"):
            termination_event.set()
            workerA.join(timeout=5)
            if workerA.is_alive():
                logging.info("Worker A did not terminate in time. Terminating forcefully.")
                workerA.terminate()
                workerA.join()
            shm.close()
            shm.unlink()
            cap.release()
            
            # Print the total number of frames processed.
            logging.info(f"Total results stored: {len(results_dict)}")

if __name__ == "__main__":
    with timer("server"):
        server()
