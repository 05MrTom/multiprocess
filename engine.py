# engine_blocking.py
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from court import read_court
from pose import YoloPoseDetector

PROCESS_THRESHOLD = 5
BUFFER_SIZE = 1  # Use a buffer size of 1 for immediate processing

def worker_a(detector, shm_name, frame_shape, frame_dtype, free_slots, frame_queue, trigger_event, termination_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray((BUFFER_SIZE,) + frame_shape, dtype=frame_dtype, buffer=shm.buf)
    
    processed_count = 0
    frame_counter = 0
    while True:
        if termination_event.is_set() and frame_queue.empty():
            print("[Worker A] Termination signaled and no frames left. Exiting.")
            break
        
        try:
            idx = frame_queue.get(timeout=1)
        except:
            continue
        
        frame = buffer[idx].copy()
        print(f"[Worker A] Processing frame {frame_counter} from buffer index {idx}")
        detector.frame_added(frame_counter, frame)
        print("df_top length:", len(detector.df_top))
        print("df_bottom length:", len(detector.df_bottom))
        
        processed_count += 1
        frame_counter += 1
        
        free_slots.put(idx)
        
        if processed_count % PROCESS_THRESHOLD == 0:
            print(f"[Worker A] Processed {processed_count} frames; triggering Worker B")
            trigger_event.set()
    
    shm.close()

def worker_b(trigger_event, termination_event):
    while not termination_event.is_set():
        if trigger_event.wait(timeout=1):
            print("[Worker B] Trigger event received! Starting specialized processing...")
            trigger_event.clear()

def server():
    court_pts = read_court("/home/amrit05/.shuttlengine/court.out")
    detector = YoloPoseDetector(court_pts)
    
    video_path = "/home/amrit05/.shuttlengine/rally_101047_101584.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Server] Failed to open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("[Server] Failed to read the first frame from video.")
        return
    frame_shape = frame.shape
    frame_dtype = frame.dtype
    
    frame_size = frame.nbytes
    total_size = frame_size * BUFFER_SIZE
    shm = shared_memory.SharedMemory(create=True, size=total_size)
    buffer = np.ndarray((BUFFER_SIZE,) + frame_shape, dtype=frame_dtype, buffer=shm.buf)
    buffer[:] = 0
    
    free_slots = mp.Queue(maxsize=BUFFER_SIZE)
    frame_queue = mp.Queue(maxsize=BUFFER_SIZE)
    
    for i in range(BUFFER_SIZE):
        free_slots.put(i)
    
    trigger_event = mp.Event()
    termination_event = mp.Event()
    
    workerA = mp.Process(target=worker_a, args=(
        detector, shm.name, frame_shape, frame_dtype,
        free_slots, frame_queue, trigger_event, termination_event
    ), name="Worker-A")
    workerB = mp.Process(target=worker_b, args=(trigger_event, termination_event), name="Worker-B")
    workerA.start()
    workerB.start()
    
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
            
            idx = free_slots.get()  # This will block if no free slot is available.
            buffer[idx] = frame
            print(f"[Server] Frame {frame_counter} written to buffer index {idx}")
            frame_counter += 1
            
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
        workerB.terminate()
        workerB.join()
        shm.close()
        shm.unlink()
        cap.release()
        print("[Server] Shutdown complete.")

if __name__ == "__main__":
    server()
