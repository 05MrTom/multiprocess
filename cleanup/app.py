import argparse
import torch
import cv2  # For image/video loading
import numpy as np
from .predict import YoloPredictor
from contextlib import contextmanager
import time
import logging


LOGGER = logging.getLogger(__name__)

# Set up logging to log to both console and a file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(log_formatter)

# Add both handlers to the logger
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)
LOGGER.setLevel(logging.INFO)


@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000  # in ms
    elapsed = (time.perf_counter() - start) * 1000
    LOGGER.info(f"{name} took {elapsed:.2f} ms")

# Assuming the YoloPredictor class is defined elsewhere

def process_image(image_path, predictor):
    # Load the image as a NumPy array (using OpenCV)
    image_np = cv2.imread(image_path)  # Reading the image as NumPy array
    if image_np is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Perform inference on the image
    with timer("Inference"):
        results = predictor.predict(image_np)  # Pass the NumPy array

    # LOGGER.info(f"Results: {results}")

def process_video(video_path, predictor):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame
        with timer("Inference"):
            results = predictor.predict(frame)  # Pass the NumPy array (frame)

        # LOGGER.info(f"Results: {results}")

        # Optionally, display the frame with bounding boxes
        # cv2.imshow("Frame", frame)  # Uncomment if you want to see the video frames

        # Press 'q' to exit the video preview (optional)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()  # Uncomment if you show video frames

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Run YoloPredictor model")

    # Command-line arguments for engine path, device, half precision, and image/video path
    parser.add_argument('--engine', type=str, required=True, help='Path to the engine file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to run model on (default: cpu)')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16) for inference')
    parser.add_argument('--image', type=str, help='Path to the input image (for image input)')
    parser.add_argument('--video', type=str, help='Path to the input video (for video input)')

    # Parse the arguments
    args = parser.parse_args()

    # Set up the device based on argument
    device = torch.device(args.device)

    # Instantiate YoloPredictor class
    with timer("Model loading"):
        predictor = YoloPredictor(engine_path=args.engine, device=device, half=args.half)

    # Process either image or video
    if args.image:
        process_image(args.image, predictor)
    elif args.video:
        process_video(args.video, predictor)
    else:
        print("Error: You must specify either an image or video input.")
        return

if __name__ == "__main__":
    main()
