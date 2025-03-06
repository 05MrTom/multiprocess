# # engine.py
# import multiprocessing as mp
# from multiprocessing import shared_memory
# import numpy as np
# import time
# import cv2
# from contextlib import contextmanager
# import logging

# from court import read_court
# from pose import YoloPoseDetector
# from common import SharedFrameStore  # Import the new shared frame store

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# @contextmanager
# def timer(name=""):
#     start = time.perf_counter()
#     yield lambda: time.perf_counter() - start
#     elapsed = time.perf_counter() - start
#     logging.info(f"{name} took {elapsed:.4f} seconds")

# def worker_a(detector, shm_name, max_frames, frame_shape, frame_dtype,
#              new_frame_event, frame_processed_event,
#              termination_event, results_dict,
#              write_index, lock):
#     """
#     Worker A:
#       - Attaches to the shared ring buffer.
#       - Waits for new_frame_event and then reads the most recently enqueued frame.
#       - Runs pose inference and stores results.
#       - Signals processing completion.
#     """
#     # Attach to the shared memory block.
#     shm = shared_memory.SharedMemory(name=shm_name)
#     buffer = np.ndarray((max_frames, *frame_shape), dtype=frame_dtype, buffer=shm.buf)
    
#     local_frame_id = 0
#     while True:
#         if termination_event.is_set():
#             logging.info("Termination signaled. Exiting worker.")
#             break

#         # Wait for a new frame signal.
#         new_frame_event.wait()
#         new_frame_event.clear()

#         # Retrieve the latest frame from the ring buffer.
#         with lock:
#             # Compute the index of the most recent frame.
#             idx = (write_index.value - 1) % max_frames
#         frame = buffer[idx].copy()
#         logging.info(f"Worker processing frame {local_frame_id} from index {idx}")
        
#         with timer("detector.frame_added"):
#             detector.frame_added(local_frame_id, frame)
        
#         # Extract keypoints from detector's DataFrames.
#         if not detector.df_top.empty and not detector.df_bottom.empty:
#             top_keypoints = detector.df_top.iloc[-1].tolist()
#             bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
#         else:
#             top_keypoints = []
#             bottom_keypoints = []
        
#         results_dict[local_frame_id] = {"top": top_keypoints, "bottom": bottom_keypoints}
#         logging.info(f"Stored results for frame {local_frame_id}")
#         local_frame_id += 1

#         # Signal that processing is complete.
#         frame_processed_event.set()
    
#     shm.close()

# def server():
#     """
#     Server:
#       - Opens the video and creates a shared frame store to hold a ring-buffer of frames.
#       - Uses Events for synchronization between the server and worker.
#       - Enqueues each frame into the shared frame store and signals Worker A.
#     """
#     # Initialize the YOLO pose detector.
#     court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
#     detector = YoloPoseDetector(court_pts)
    
#     video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"  # Replace with your video path.
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logging.error(f"Failed to open video: {video_path}")
#         return
    
#     ret, frame = cap.read()
#     if not ret:
#         logging.error("Failed to read the first frame from video.")
#         return
#     frame_shape = frame.shape       # e.g., (height, width, channels)
#     frame_dtype = frame.dtype
    
#     # Create a SharedFrameStore to hold multiple frames.
#     max_frames = 10  # Fixed capacity for the ring buffer.
#     shared_store = SharedFrameStore(max_frames, frame_shape, frame_dtype)
    
#     manager = mp.Manager()
#     results_dict = manager.dict()
    
#     # Create Events for synchronization.
#     new_frame_event = mp.Event()         # Set by the server when a new frame is enqueued.
#     frame_processed_event = mp.Event()   # Set by Worker A when processing is complete.
#     termination_event = mp.Event()       # Signal termination.
    
#     # Initially, signal that no frame is currently being processed.
#     frame_processed_event.set()
    
#     # Start the worker process.
#     workerA = mp.Process(target=worker_a, args=(
#         detector,
#         shared_store.shm.name,
#         max_frames,
#         frame_shape,
#         frame_dtype,
#         new_frame_event,
#         frame_processed_event,
#         termination_event,
#         results_dict,
#         shared_store.write_index,       shared_store.lock
#     ), name="Worker-A")
#     workerA.start()
    
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     frame_interval = 1.0 / fps
#     frame_counter = 0
#     try:
#         while True:
#             start_time = time.perf_counter()
            
#             # Wait for the previous frame to be processed.
#             frame_processed_event.wait()
#             frame_processed_event.clear()
            
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("End of video reached.")
#                 break
            
#             # Enqueue the new frame into the shared frame store.
#             shared_store.enqueue(frame)
#             logging.info(f"Enqueued frame {frame_counter} into shared frame store.")
#             frame_counter += 1
            
#             # Signal Worker A that a new frame is available.
#             new_frame_event.set()
            
#             elapsed_time = time.perf_counter() - start_time
#             time_to_wait = frame_interval - elapsed_time
#             if time_to_wait > 0:
#                 time.sleep(time_to_wait)
#     except KeyboardInterrupt:
#         logging.error("KeyboardInterrupt, shutting down.")
#     finally:
#         termination_event.set()
#         workerA.join(timeout=5)
#         if workerA.is_alive():
#             logging.info("Worker A did not terminate in time. Terminating forcefully.")
#             workerA.terminate()
#             workerA.join()
#         shared_store.cleanup()
#         cap.release()
        
#         # Print the total number of frames processed.
#         logging.info(f"Total results stored: {len(results_dict)}")

# if __name__ == "__main__":
#     server()


# engine.py
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from contextlib import contextmanager
import logging

from court import read_court
from pose import YoloPoseDetector
from posel import YoloPoseDetectorL
from common import SharedFrameStore  # Import the new shared frame store

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

def worker_a(detector, shm_name, max_frames, frame_shape, frame_dtype,
             new_frame_event, frame_processed_event,
             termination_event, results_dict,
             frame_id, write_index, lock):
    """
    Worker A:
      - Attaches to the shared ring buffer.
      - Waits for new_frame_event and then reads the most recently enqueued frame.
      - Runs pose inference and stores results under keys "top" and "bottom".
    """
    # Attach to the shared memory block.
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((max_frames, *frame_shape), dtype=frame_dtype, buffer=shm.buf)
    
    while True:
        if termination_event.is_set():
            logging.info("Termination signaled. Exiting worker A.")
            break

        # Wait for a new frame signal.
        new_frame_event.wait()
        new_frame_event.clear()

        # Retrieve the latest frame from the ring buffer.
        with lock:
            idx = (write_index.value - 1) % max_frames
        frame = buffer[idx].copy()
        current_frame_id = frame_id.value  # use the shared frame id
        
        logging.info(f"Worker A processing frame {current_frame_id} from index {idx}")
        
        with timer(f"detector.worker_a_processing {current_frame_id}"):
            # Run your detector processing
            detector.frame_added(current_frame_id, frame)
        # Extract keypoints from detector's DataFrames.
        if not detector.df_top.empty and not detector.df_bottom.empty:
            top_keypoints = detector.df_top.iloc[-1].tolist()
            bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top_keypoints = []
            bottom_keypoints = []
        
        # Store the results (for this frame, create a new dict entry)
        results_dict[current_frame_id] = {"top": top_keypoints, "bottom": bottom_keypoints}
        logging.info(f"Worker A stored results for frame {current_frame_id}")
        
        # Signal that processing is complete.
        frame_processed_event.set()
    
    shm.close()

def worker_b(detector, shm_name, max_frames, frame_shape, frame_dtype,
             new_frame_event, frame_processed_event,
             termination_event, results_dict,
             frame_id, write_index, lock):
    """
    Worker B:
      - Attaches to the shared ring buffer.
      - Waits for its new_frame_event and then reads the most recently enqueued frame.
      - Performs an alternative processing and updates the existing dict entry
        by merging in results under keys "top2" and "bottom2".
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((max_frames, *frame_shape), dtype=frame_dtype, buffer=shm.buf)
    
    while True:
        if termination_event.is_set():
            logging.info("Termination signaled. Exiting worker B.")
            break

        new_frame_event.wait()
        new_frame_event.clear()

        # Retrieve the latest frame from the ring buffer.
        with lock:
            idx = (write_index.value - 1) % max_frames
        frame = buffer[idx].copy()
        current_frame_id = frame_id.value  # use the shared frame id
        
        logging.info(f"Worker B processing frame {current_frame_id} from index {idx}")
        
        with timer(f"detector.worker_b_processing {current_frame_id}"):
            # Run your detector processing
            detector.frame_added(current_frame_id, frame)
        # Here you might call a different processing method.
        # We use the same detector for demonstration.
        if not detector.df_top.empty and not detector.df_bottom.empty:
            top2_keypoints = detector.df_top.iloc[-1].tolist()
            bottom2_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top2_keypoints = []
            bottom2_keypoints = []
        
        # Merge Worker B's results with any existing entry in results_dict.
        # (We use "bottom2" to avoid overwriting Worker A's "bottom".)
        # If Worker A hasn’t stored the results yet, create a new entry.
        temp = results_dict.get(current_frame_id, {})
        temp.update({"top2": top2_keypoints, "bottom2": bottom2_keypoints})
        results_dict[current_frame_id] = temp
        logging.info(f"Worker B stored results for frame {current_frame_id}")
        
        # Signal that processing is complete.
        frame_processed_event.set()
    
    shm.close()

def server():
    """
    Server:
      - Opens the video and creates a shared frame store (a ring buffer).
      - Uses Events for synchronization between the server and both workers.
      - Enqueues each frame into the shared frame store and signals both workers.
    """
    # Initialize the YOLO pose detector.
    court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
    detector = YoloPoseDetector(court_pts)
    detectorl = YoloPoseDetectorL(court_pts)
    
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
    
    # Create a SharedFrameStore to hold multiple frames.
    max_frames = 10  # Fixed capacity for the ring buffer.
    shared_store = SharedFrameStore(max_frames, frame_shape, frame_dtype)
    
    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Create a shared frame id (so both workers process the same frame id)
    frame_id = mp.Value('i', 0)
    
    # Create Events for synchronization.
    new_frame_event_a = mp.Event()         # For Worker A
    frame_processed_event_a = mp.Event()   # For Worker A
    new_frame_event_b = mp.Event()         # For Worker B
    frame_processed_event_b = mp.Event()   # For Worker B
    termination_event = mp.Event()         # Signal termination.
    
    # Initially, signal that no frame is currently being processed.
    frame_processed_event_a.set()
    frame_processed_event_b.set()
    
    # Start the worker processes.
    workerA = mp.Process(target=worker_a, args=(
        detector,
        shared_store.shm.name,
        max_frames,
        frame_shape,
        frame_dtype,
        new_frame_event_a,
        frame_processed_event_a,
        termination_event,
        results_dict,
        frame_id,
        shared_store.write_index,
        shared_store.lock
    ), name="Worker-A")
    workerA.start()
    
    workerB = mp.Process(target=worker_b, args=(
        detectorl,
        shared_store.shm.name,
        max_frames,
        frame_shape,
        frame_dtype,
        new_frame_event_b,
        frame_processed_event_b,
        termination_event,
        results_dict,
        frame_id,
        shared_store.write_index,
        shared_store.lock
    ), name="Worker-B")
    workerB.start()
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # Wait for both workers to finish processing the previous frame.
            frame_processed_event_a.wait()
            frame_processed_event_a.clear()
            frame_processed_event_b.wait()
            frame_processed_event_b.clear()
            
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video reached.")
                break
            
            # Update the shared frame id.
            with frame_id.get_lock():
                frame_id.value += 1
                current_frame_id = frame_id.value
            
            # Enqueue the new frame into the shared frame store.
            with timer("shared store"):
                shared_store.enqueue(frame)
            logging.info(f"Enqueued frame {current_frame_id} into shared frame store.")
            
            # Signal both workers that a new frame is available.
            new_frame_event_a.set()
            new_frame_event_b.set()
            
            elapsed_time = time.perf_counter() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt, shutting down.")
    finally:
        termination_event.set()
        workerA.join(timeout=5)
        workerB.join(timeout=5)
        if workerA.is_alive():
            logging.warning("Worker A did not terminate in time. Terminating forcefully.")
            workerA.terminate()
            workerA.join()
        if workerB.is_alive():
            logging.warning("Worker B did not terminate in time. Terminating forcefully.")
            workerB.terminate()
            workerB.join()
        shared_store.cleanup()
        cap.release()
        
        # Print the total number of frames processed.
        logging.info(f"Total results stored: {len(results_dict)}")
        # logging.info(f"last frame dict: {results_dict[current_frame_id]}")
        # logging.info(f"first frame dict: {results_dict[1]}")

if __name__ == "__main__":
    with timer("server"):
        server()
