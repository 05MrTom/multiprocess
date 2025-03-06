# # engine_shared_no_buffer.py
# import multiprocessing as mp
# from multiprocessing import shared_memory
# import numpy as np
# import time
# import cv2
# from contextlib import contextmanager

# from court import read_court
# from pose import YoloPoseDetector

# @contextmanager
# def timer(name=""):
#     start = time.perf_counter()
#     yield lambda: time.perf_counter() - start
#     elapsed = time.perf_counter() - start
#     print(f"{name} took {elapsed:.4f} seconds")

# def worker_a(detector, shm_name, frame_shape, frame_dtype,
#              new_frame_event, frame_processed_event,
#              termination_event, results_dict):
#     """
#     Worker A:
#       - Waits for a new frame signal.
#       - Reads the frame from shared memory.
#       - Runs pose inference using detector.frame_added.
#       - Extracts keypoints for top and bottom players.
#       - Stores the results in the Manager dictionary keyed by frame id.
#       - Signals back that the frame has been processed.
#     """
#     # Attach to the shared memory block.
#     shm = shared_memory.SharedMemory(name=shm_name)
#     # Create a NumPy view of the shared memory (one frame only).
#     frame_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
#     frame_id = 0
#     while True:
#         # Exit if termination is signaled.
#         if termination_event.is_set():
#             print("Termination signaled. Exiting.")
#             break
        
#         # Wait for a new frame to be written.
#         new_frame_event.wait()
#         new_frame_event.clear()
        
#         # Copy the frame from shared memory.
#         frame = frame_array.copy()
#         print(f"Processing frame {frame_id}")
        
#         with timer("detector.frame_added"):
#             detector.frame_added(frame_id, frame)
        
#         # Extract keypoints from detector's DataFrames.
#         if not detector.df_top.empty and not detector.df_bottom.empty:
#             top_keypoints = detector.df_top.iloc[-1].tolist()
#             bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
#         else:
#             top_keypoints = []
#             bottom_keypoints = []
        
#         # Store the results in the Manager dictionary.
#         results_dict[frame_id] = {"top": top_keypoints, "bottom": bottom_keypoints}
#         print(f"Stored results for frame {frame_id}")
#         frame_id += 1
        
#         # Signal back that processing is complete.
#         frame_processed_event.set()
        
#     shm.close()

# def server():
#     """
#     Server:
#       - Opens the video.
#       - Allocates shared memory to hold a single frame.
#       - Creates a Manager dictionary to store processed keypoints keyed by frame id.
#       - Uses two Events to synchronize with Worker A:
#             * new_frame_event: signals that a new frame is available.
#             * frame_processed_event: signals that the frame has been processed.
#       - Reads frames from the video, writes them into shared memory,
#         and signals Worker A. Then waits until Worker A processes the frame.
#     """
#     # Initialize the YOLO pose detector.
#     court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
#     detector = YoloPoseDetector(court_pts)
    
#     video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"  # Replace with your video path.
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_path}")
#         return
    
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to read the first frame from video.")
#         return
#     frame_shape = frame.shape       # e.g., (height, width, channels)
#     frame_dtype = frame.dtype
    
#     # Create shared memory to hold one frame.
#     frame_size = frame.nbytes
#     shm = shared_memory.SharedMemory(create=True, size=frame_size)
#     # Create a NumPy array view of the shared memory.
#     frame_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
#     # Create a Manager dictionary for storing keypoints.
#     manager = mp.Manager()
#     results_dict = manager.dict()
    
#     # Create synchronization Events.
#     new_frame_event = mp.Event()         # Set by server when a new frame is written.
#     frame_processed_event = mp.Event()   # Set by worker when processing is complete.
#     termination_event = mp.Event()       # Signaled to terminate processing.
    
#     # Initially, signal that no frame is being processed.
#     frame_processed_event.set()
    
#     # Start Worker A.
#     workerA = mp.Process(target=worker_a, args=(
#         detector, shm.name, frame_shape, frame_dtype,
#         new_frame_event, frame_processed_event,
#         termination_event, results_dict
#     ), name="Worker-A")
#     workerA.start()
    
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     frame_interval = 1.0 / fps
#     frame_counter = 0
#     try:
#         while True:
#             start_time = time.perf_counter()
#             # Wait until the previous frame has been processed.
#             frame_processed_event.wait()
#             frame_processed_event.clear()
            
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video reached.")
#                 break
            
#             # Write the new frame into shared memory.
#             np.copyto(frame_array, frame)
#             print(f"Wrote frame {frame_counter} to shared memory.")
#             frame_counter += 1
            
#             # Signal Worker A that a new frame is available.
#             new_frame_event.set()
            
#             elapsed_time = time.perf_counter() - start_time
#             time_to_wait = frame_interval - elapsed_time
#             if time_to_wait > 0:
#                 time.sleep(time_to_wait)
#     except KeyboardInterrupt:
#         print("KeyboardInterrupt, shutting down.")
#     finally:
#         termination_event.set()
#         workerA.join(timeout=5)
#         if workerA.is_alive():
#             print("Worker A did not terminate in time. Terminating forcefully.")
#             workerA.terminate()
#             workerA.join()
#         shm.close()
#         shm.unlink()
#         cap.release()
        
#         # Optionally, print the results stored in the Manager dictionary.
#         print(f"Total results stored: {len(results_dict)}")


# if __name__ == "__main__":
#     server()


# engine_shared_no_buffer_event.py
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

def worker_a(detector, shm_name, frame_shape, frame_dtype,
             new_frame_event, frame_processed_event,
             termination_event, results_dict):
    """
    Worker A:
      - Waits (blocking) for the new_frame_event.
      - Reads the frame from shared memory.
      - Runs pose inference (via detector.frame_added).
      - Extracts keypoints for top and bottom players.
      - Stores the results in the Manager dictionary keyed by frame id.
      - Signals back with frame_processed_event so the server can proceed.
    """
    # Attach to the shared memory block.
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
    frame_id = 0
    while True:
        # Exit if termination is signaled.
        if termination_event.is_set():
            logging.info("Termination signaled. Exiting.")
            break
        
        # Wait for the server to signal that a new frame is ready.
        new_frame_event.wait()
        new_frame_event.clear()
        
        # Copy the frame from shared memory.
        frame = frame_array.copy()
        logging.info(f"Processing frame {frame_id}")
        
        with timer("detector.frame_added"):
            detector.frame_added(frame_id, frame)
        
        # Extract keypoints from detector's DataFrames.
        if not detector.df_top.empty and not detector.df_bottom.empty:
            top_keypoints = detector.df_top.iloc[-1].tolist()
            bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top_keypoints = []
            bottom_keypoints = []
        
        # Store the results using frame_id as the key.
        results_dict[frame_id] = {"top": top_keypoints, "bottom": bottom_keypoints}
        logging.info(f"Stored results for frame {frame_id}")
        
        frame_id += 1
        
        # Signal that processing of the current frame is complete.
        frame_processed_event.set()
    
    shm.close()

def server():
    """
    Server:
      - Opens the video and creates shared memory to hold one frame.
      - Uses two Events for synchronization:
            * new_frame_event: set when a new frame is written.
            * frame_processed_event: set when Worker A has processed the frame.
      - Uses a Manager dictionary to store results keyed by frame id.
      - For each frame read from the video, writes it into shared memory, triggers Worker A,
        and waits for processing to complete before moving on.
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
    
    # Create shared memory to hold a single frame.
    frame_size = frame.nbytes
    shm = shared_memory.SharedMemory(create=True, size=frame_size)
    frame_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Create Events for synchronization.
    new_frame_event = mp.Event()         # Server sets this after writing a frame.
    frame_processed_event = mp.Event()   # Worker A sets this after processing a frame.
    termination_event = mp.Event()       # Signal termination.
    
    # Initially, signal that no frame is currently being processed.
    frame_processed_event.set()
    
    workerA = mp.Process(target=worker_a, args=(
        detector, shm.name, frame_shape, frame_dtype,
        new_frame_event, frame_processed_event,
        termination_event, results_dict
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
            
            # Write the new frame into shared memory.
            np.copyto(frame_array, frame)
            logging.info(f"Wrote frame {frame_counter} to shared memory.")
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
    server()
