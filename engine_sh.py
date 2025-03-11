# engine_shared_manager.py
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from court import read_court
from pose import YoloPoseDetector

from contextlib import contextmanager
import time

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed:.4f} seconds")


PROCESS_THRESHOLD = 5
BUFFER_SIZE = 1  # Number of frame slots in shared memory

def worker_a(detector, shm_name, frame_shape, frame_dtype,
             free_slots, frame_queue, 
             termination_event, results_dict):
    """
    Worker A:
      - Retrieves a buffer index from frame_queue.
      - Reads the frame from shared memory.
      - Runs pose inference (via detector.frame_added).
      - Extracts keypoints for top and bottom players.
      - Stores the results in the Manager dictionary using the frame id as key.
      - Returns the slot index to free_slots.
    """
    # Attach to the shared memory block.
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((BUFFER_SIZE,) + frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
    processed_count = 0
    frame_id = 0
    while True:
        # Exit if termination is signaled and no frames remain in the queue.
        if termination_event.is_set() and frame_queue.empty():
            print("[Worker A] Termination signaled and no frames left. Exiting.")
            break
        
        try:
            idx = frame_queue.get(timeout=1)
        except Exception as e:
            continue  # No frame index available, check termination
        
        # Get the frame from shared memory.
        frame = buffer[idx].copy()
        print(f"[Worker A] Processing frame {frame_id} from buffer index {idx}")
        
        # Run pose inference on the frame.
        with timer("detector.frame_added"):
            detector.frame_added(frame_id, frame)
        
        # Extract keypoints from the detector's DataFrames.
        # (Assuming each call to frame_added appends one row for top and bottom.)
        if not detector.df_top.empty and not detector.df_bottom.empty:
            # Take the last row (as list) from each DataFrame.
            top_keypoints = detector.df_top.iloc[-1].tolist()   # List[List[float]] if structured
            bottom_keypoints = detector.df_bottom.iloc[-1].tolist()
        else:
            top_keypoints = []
            bottom_keypoints = []
        
        # Store the results in the Manager dictionary.
        results_dict[frame_id] = {"top": top_keypoints, "bottom": bottom_keypoints}
        print(f"[Worker A] Stored results for frame {frame_id} in manager dict.")
        
        processed_count += 1
        frame_id += 1
        
        # Return the buffer index back to free_slots.
        free_slots.put(idx)
           
    shm.close()


def server():
    """
    Server:
      - Opens the video and preallocates shared memory for BUFFER_SIZE frames.
      - Initializes two queues:
          * free_slots: holds available buffer indices.
          * frame_queue: holds indices for frames ready to be processed.
      - Creates a Manager dictionary to store results (keypoints) keyed by frame id.
      - Spawns Worker A (for pose inference) and Worker B (for additional processing).
      - Reads frames from the video, writes them into shared memory, and enqueues the index.
    """
    # Initialize the YOLO pose detector.
    court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
    detector = YoloPoseDetector(court_pts)
    
    video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"  # Replace with your video path.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Server] Failed to open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("[Server] Failed to read the first frame from video.")
        return
    frame_shape = frame.shape       # e.g., (height, width, channels)
    frame_dtype = frame.dtype
    
    # Create shared memory for BUFFER_SIZE frames.
    frame_size = frame.nbytes
    total_size = frame_size * BUFFER_SIZE
    shm = shared_memory.SharedMemory(create=True, size=total_size)
    buffer = np.ndarray((BUFFER_SIZE,) + frame_shape, dtype=frame_dtype, buffer=shm.buf)
    buffer[:] = 0  # Initialize to zeros.
    
    # Create two queues: free_slots and frame_queue.
    free_slots = mp.Queue(maxsize=BUFFER_SIZE)
    frame_queue = mp.Queue(maxsize=BUFFER_SIZE)
    for i in range(BUFFER_SIZE):
        free_slots.put(i)
    
    # Create a Manager dictionary to store keypoints by frame id.
    manager = mp.Manager()
    results_dict = manager.dict()
    
    trigger_event = mp.Event()
    termination_event = mp.Event()
    
    # Start Worker A and Worker B.
    workerA = mp.Process(target=worker_a, args=(
        detector, shm.name, frame_shape, frame_dtype,
        free_slots, frame_queue, termination_event, results_dict
    ), name="Worker-A")
    workerA.start()
    
    # Get video FPS and calculate frame interval.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps
    frame_counter = 0
    try:
        while True:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[Server] End of video reached.")
                break
            
            # Get a free buffer index (this blocks if none are available).
            idx = free_slots.get()
            buffer[idx] = frame
            print(f"[Server] Frame {frame_counter} written to buffer index {idx}")
            frame_counter += 1
            
            # Enqueue the buffer index for processing.
            frame_queue.put(idx)
            
            elapsed_time = time.perf_counter() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
    except KeyboardInterrupt:
        print("[Server] KeyboardInterrupt, shutting down.")
    finally:
        termination_event.set()
        workerA.join()
        shm.close()
        shm.unlink()
        cap.release()
        
        # # Optionally, print or save the results from the manager dict.
        # for fid, data in sorted(results_dict.items()):
        #     print(f"Frame {fid}: {data}")
        print(f"len(results_dict): {len(results_dict)}")

if __name__ == "__main__":
    server()
