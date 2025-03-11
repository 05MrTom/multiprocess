import multiprocessing as mp
import numpy as np
import time
import cv2
from contextlib import contextmanager
import logging

from court import read_court
from pose import YoloPoseDetector

# Import ShareableList from the shared_memory module.
from multiprocessing.shared_memory import ShareableList

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

def worker_a(detector, shareable_list_name, frame_shape, frame_dtype,
             new_frame_event, frame_processed_event,
             termination_event, results_dict, current_index, buffer_size,
             expected_length):
    """
    Worker A:
      - Reopens the ShareableList by name.
      - Waits for new_frame_event.
      - Reads the most recently written frame (from the circular buffer).
      - Converts the stored bytes back to a NumPy array.
      - Runs the pose detector and stores results.
      - Signals when processing is complete.
    """
    # Reopen the shared list using its name.
    sl = ShareableList(name=shareable_list_name)
    
    processed_frame = 0
    while not termination_event.is_set():
        new_frame_event.wait()
        new_frame_event.clear()
        
        with current_index.get_lock():
            # The latest written frame is at (current_index - 1) mod buffer_size.
            idx = (current_index.value - 1) % buffer_size
        
        frame_bytes = sl[idx]
        if not frame_bytes:
            logging.warning("Worker: No frame available at index %d", idx)
            continue
        
        # Verify the length of the bytes and log if there's a mismatch.
        if len(frame_bytes) != expected_length:
            logging.warning("Worker: Frame bytes length %d does not match expected %d",
                            len(frame_bytes), expected_length)
        
        # Convert the bytes back to a NumPy array.
        try:
            frame = np.frombuffer(frame_bytes, dtype=frame_dtype).reshape(frame_shape)
        except ValueError as e:
            logging.error("Worker: Reshape error: %s", e)
            continue
        
        logging.info(f"Worker: Processing frame from shareable list at index {idx}")
        
        with timer("detector.frame_added"):
            detector.frame_added(processed_frame, frame)
        
        # Extract keypoints from the detector’s DataFrames.
        if not detector.df_top.empty and not detector.df_bottom.empty:
            top_keypoints = detector.df_top.iloc[-1].tolist()
            bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top_keypoints = []
            bottom_keypoints = []
        
        results_dict[processed_frame] = {"top": top_keypoints, "bottom": bottom_keypoints}
        logging.info(f"Worker: Stored results for frame {processed_frame}")
        
        processed_frame += 1
        frame_processed_event.set()
    
    sl.shm.close()

def server():
    """
    Server:
      - Opens the video and creates a ShareableList as a circular buffer (size 30) to hold images (as bytes).
      - For each frame, converts it to bytes and writes it into the circular buffer.
      - Signals Worker A to process the frame.
      - Uses events to synchronize frame processing.
    """
    # Initialize the YOLO pose detector.
    court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
    detector = YoloPoseDetector(court_pts)
    
    video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video: %s", video_path)
        return
    
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read the first frame from video.")
        return
    frame_shape = frame.shape       # e.g., (height, width, channels)
    frame_dtype = frame.dtype
    
    buffer_size = 30
    # Preallocate each slot with a bytes object of length equal to frame.nbytes.
    frame_bytes_length = frame.nbytes
    expected_length = frame_bytes_length
    sl = ShareableList([b'\x00' * frame_bytes_length for _ in range(buffer_size)])
    
    # Shared index to track the next write position.
    current_index = mp.Value('i', 0)
    
    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Create synchronization events.
    new_frame_event = mp.Event()         # Signaled when a new frame is written.
    frame_processed_event = mp.Event()   # Signaled when Worker A finishes processing.
    termination_event = mp.Event()       # Signals termination.
    
    # Initially, signal that processing is done.
    frame_processed_event.set()
    
    workerA = mp.Process(target=worker_a, args=(
        detector, sl.shm.name, frame_shape, frame_dtype,
        new_frame_event, frame_processed_event,
        termination_event, results_dict, current_index, buffer_size,
        expected_length
    ), name="Worker-A")
    workerA.start()
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps
    frame_counter = 0
    try:
        while True:
            start_time = time.perf_counter()
            
            # Wait for the previous frame to be processed.
            frame_processed_event.wait()
            frame_processed_event.clear()
            
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video reached.")
                break
            
            # Convert the frame to bytes.
            frame_bytes = frame.tobytes()
            # Check if the frame bytes length matches the expected length.
            if len(frame_bytes) != expected_length:
                logging.warning("Server: Frame bytes length %d does not match expected %d. Padding.",
                                len(frame_bytes), expected_length)
                frame_bytes = frame_bytes.ljust(expected_length, b'\x00')
            
            with current_index.get_lock():
                idx = current_index.value
                sl[idx] = frame_bytes
                current_index.value = (idx + 1) % buffer_size
            logging.info(f"Server: Wrote frame {frame_counter} to shareable list index {idx}")
            frame_counter += 1
            
            # Signal Worker A that a new frame is available.
            new_frame_event.set()
            
            elapsed_time = time.perf_counter() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt, shutting down.")
    finally:
        termination_event.set()
        workerA.join(timeout=5)
        if workerA.is_alive():
            logging.info("Worker A did not terminate in time. Terminating forcefully.")
            workerA.terminate()
            workerA.join()
        sl.shm.close()
        sl.shm.unlink()
        cap.release()
        
        logging.info("Total results stored: %d", len(results_dict))

if __name__ == "__main__":
    server()
