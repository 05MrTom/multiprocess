import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import random
import time

# Threshold at which Child Worker A triggers Child Worker B
PROCESS_THRESHOLD = 5

def child_worker_a(shm_name, data_event, trigger_event):
    """
    Child Worker A:
      - Blocks until new data is signaled by the parent.
      - Reads an integer from shared memory.
      - Maintains a count; when a threshold is reached, triggers trigger_event.
    """
    # Attach to the existing shared memory block and create a NumPy view
    shm = shared_memory.SharedMemory(name=shm_name)
    # Let's assume the shared memory holds a single 32-bit integer.
    data_array = np.ndarray((1,), dtype=np.int32, buffer=shm.buf)
    
    processed_count = 0
    while True:
        # Block until the parent signals new data is available.
        data_event.wait()  
        # Read the data
        new_data = int(data_array[0])
        print(f"[Worker A] Received data: {new_data}")
        processed_count += 1

        # (Your real processing logic would go here)

        # Clear the data event so we can wait for the next data item.
        # (This assumes all Worker A processes share the same event.
        # In more complex scenarios, you might use individual events.)
        data_event.clear()

        # If we've processed a threshold number of items, trigger Worker B.
        if processed_count % PROCESS_THRESHOLD == 0:
            print(f"[Worker A] Processed {processed_count} items; triggering Worker B")
            trigger_event.set()


def child_worker_b(trigger_event):
    """
    Child Worker B:
      - Waits (blocking) until triggered by Worker A.
      - Performs specialized processing when the trigger event is signaled.
    """
    while True:
        # Block until triggered by Worker A.
        trigger_event.wait()
        print("[Worker B] Trigger event received! Starting specialized processing...")
        
        # (Perform your specialized processing here.)

        # Clear the trigger event so Worker B can wait for the next trigger.
        trigger_event.clear()


def server():
    """
    Parent server:
      - Creates a shared memory block.
      - Creates two shared events: one for notifying new data, and one for inter-child communication.
      - Spawns long-running child processes.
      - Simulates receiving data and writes it into shared memory, then signals Worker A.
    """
    # Create shared memory (enough for a single 32-bit integer, adjust size as needed).
    shm = shared_memory.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    data_array = np.ndarray((1,), dtype=np.int32, buffer=shm.buf)

    # Create the two events.
    # data_event is used to signal that new data has arrived.
    # trigger_event is used for inter-child communication.
    data_event = mp.Event()
    trigger_event = mp.Event()

    # Spawn child processes:
    # We'll spawn one instance of Worker A and one of Worker B.
    # In a real scenario, you might spawn multiple Worker A's; they can share the same data_event.
    worker_a = mp.Process(target=child_worker_a, args=(shm.name, data_event, trigger_event), name="Worker-A")
    worker_b = mp.Process(target=child_worker_b, args=(trigger_event,), name="Worker-B")
    worker_a.start()
    worker_b.start()
    
    counter = 0
    try:
        while True:
            # Simulate receiving new data.
            # In a real server, replace this with your actual data receipt logic.
            # Here we simulate very rapid incoming data (in milliseconds).
            time.sleep(random.uniform(0.005, 0.01))
            new_data = counter
            counter += 1
            # Write the new data into shared memory.
            data_array[0] = new_data
            print(f"[Server] New data: {new_data}")
            # Signal Worker A(s) that new data is available.
            data_event.set()
    except KeyboardInterrupt:
        print("[Server] Shutting down.")
    finally:
        worker_a.terminate()
        worker_b.terminate()
        worker_a.join()
        worker_b.join()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    server()
