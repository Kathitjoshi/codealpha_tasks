# Object Detection and Tracking

## Update / Fix Notes

Accuracy issue fixed: the script previously defaulted to the yolov8n (nano) model with no explicit NMS IoU control or inference resolution setting, which caused frequent misclassification of small/handheld objects (for example, items being identified as "cell phone" incorrectly). The default model is now yolov8s (small), with explicit imgsz=640, iou=0.45, and a raised default confidence of 0.45. New CLI flags --model, --imgsz, and --iou let you tune further (use --model yolov8m.pt for even higher accuracy if your hardware supports it).

## Overview

This project performs real-time object detection and multi-object tracking on a webcam feed or video file. It uses YOLOv8 (via the ultralytics library) for detection when available, with an automatic fallback to YOLOv3 via OpenCV's DNN module if ultralytics is not installed. Tracking is performed using a from-scratch implementation of SORT (Simple Online and Realtime Tracking), which combines a Kalman filter for motion prediction with IoU-based data association to maintain persistent object IDs across frames.

## Features

- Real-time video input from webcam or video file (OpenCV VideoCapture)
- Object detection using a pre-trained YOLO model (YOLOv8 preferred, YOLOv3 fallback)
- Frame-by-frame bounding box detection with confidence scores
- Multi-object tracking with SORT (Kalman filter motion model + IoU matching)
- Persistent track IDs that remain stable as objects move across frames
- On-screen labels showing track ID, object class, and detection confidence
- Optional output video saving
- Headless mode for running without a display (useful on servers)

## How It Works

1. Video input: Frames are read one at a time from a webcam or video file using cv2.VideoCapture.
2. Detection: Each frame is passed to the YOLO model, which returns bounding boxes, class labels, and confidence scores for detected objects.
3. Tracking: Detected bounding boxes are passed to the SORT tracker. Existing tracks predict their next position using a Kalman filter (constant velocity model). New detections are matched to predictions using Intersection over Union (IoU); a greedy matching algorithm assigns detections to the best-matching track. Unmatched detections create new tracks; tracks with no matching detection for several frames are removed.
4. Visualization: Each tracked object is drawn with a bounding box, a unique color per track ID, and a label showing the track ID, class name, and confidence.
5. Output: The annotated video stream is displayed in a window in real time and optionally saved to a file.

## Project Structure

  task4/
    object_detection_tracking.py    - Main script containing detector, SORT tracker, and video loop
    README.md                       - This file

## Requirements

Python 3.8 or higher. Install dependencies with:

    pip install opencv-python numpy ultralytics

If ultralytics cannot be installed, the script will still work using only:

    pip install opencv-python numpy

In that case, on first run the script automatically downloads the YOLOv3 configuration, class names, and weight files (approximately 236 MB) from public sources into the working directory. This requires an internet connection on first run only.

On Linux servers without a display, use opencv-python-headless instead of opencv-python:

    pip install opencv-python-headless numpy ultralytics

## How to Run

### Using a webcam

    python object_detection_tracking.py

This opens camera index 0 by default.

### Using a specific webcam index

    python object_detection_tracking.py --source 1

### Using a video file

    python object_detection_tracking.py --source path/to/video.mp4

### Saving the output to a file

    python object_detection_tracking.py --source path/to/video.mp4 --output result.mp4

### Adjusting detection confidence threshold

    python object_detection_tracking.py --source 0 --conf 0.6

### Running without a display window (headless / server mode)

    python object_detection_tracking.py --source path/to/video.mp4 --output result.mp4 --headless

In headless mode the script prints periodic progress instead of opening a window, and the annotated video is written to the output file.

### Controls

Press Q while the display window is focused to stop processing and exit.

## Command-Line Arguments

  --source     Video source: 0 (or any camera index) for webcam, or a path to a video file. Default: 0
  --output     Optional path to save the annotated output video (e.g. result.mp4). If omitted, no file is saved.
  --conf       Detection confidence threshold between 0 and 1. Default: 0.5
  --headless   Run without opening a display window. Useful when no GUI/display is available.

## Notes on Detection Backend

- If the ultralytics package is installed, the script automatically downloads the lightweight yolov8n.pt model (approximately 6 MB) on first run and uses it for detection. This is the recommended path for best speed and accuracy.
- If ultralytics is not installed or fails to load, the script falls back to YOLOv3 through OpenCV's built-in DNN module. The configuration, class list, and weights are downloaded automatically on first run if not already present in the working directory.
- Both backends detect the 80 object classes from the COCO dataset (person, car, dog, bicycle, etc).

## Notes on Tracking

- The SORT implementation in this script is self-contained (no external sort or scipy dependency required for the Hungarian algorithm; a greedy IoU-based matching is used instead).
- Each track maintains a 7-dimensional Kalman filter state (center x, center y, scale, aspect ratio, and their velocities).
- Tracks that are not matched to any detection for more than max_age frames (default 5) are removed.
- Track IDs are integers that increase monotonically as new objects appear; they are not reused after a track is dropped.

## Performance Tips

- Detection speed depends heavily on hardware. YOLOv8n on CPU typically runs at 5 to 15 FPS depending on resolution; GPU acceleration (if available and PyTorch with CUDA is installed) significantly improves throughput.
- Lowering the input video resolution or using a smaller YOLO variant improves speed at the cost of detection accuracy.
- Increasing --conf reduces false positives but may miss smaller or partially occluded objects.
