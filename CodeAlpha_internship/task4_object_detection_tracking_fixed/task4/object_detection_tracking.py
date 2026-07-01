"""
Object Detection and Tracking

Uses:
  - YOLOv8 (via ultralytics) for object detection on each video frame
  - SORT (Simple Online and Realtime Tracking) implemented from scratch
    for multi-object tracking with persistent IDs
  - OpenCV for video input (webcam or file) and display

If ultralytics is not installed, falls back to YOLOv3 via OpenCV DNN
with automatic weight download. SORT tracker works in both cases.

Usage:
  python object_detection_tracking.py                   # use webcam (camera 0)
  python object_detection_tracking.py --source video.mp4  # use video file
  python object_detection_tracking.py --source 0          # webcam index
  python object_detection_tracking.py --source video.mp4 --output out.mp4

Press Q to quit.
"""

import argparse
import sys
import os
import math
import urllib.request
import numpy as np
import cv2

# ── SORT Tracker (from scratch, no scipy needed) ─────────────────────────────

def iou(bb_test, bb_gt):
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """
    Kalman filter-based tracker for a single bounding box.
    State: [cx, cy, s, r, cx', cy', s']
      cx, cy = center x, y
      s = scale (area)
      r = aspect ratio (width/height, kept constant)
    """
    count = 0

    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Noise covariance
        self.R = np.diag([1.,1.,10.,10.])
        self.Q = np.diag([1.,1.,1.,1.,0.01,0.01,0.0001])
        self.P = np.diag([10.,10.,10.,10.,10000.,10000.,10000.])

        # Initial state
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.; cy = (y1+y2)/2.
        s = (x2-x1)*(y2-y1)
        r = (x2-x1)/(y2-y1) if (y2-y1)>0 else 1.
        self.x = np.array([[cx],[cy],[s],[r],[0.],[0.],[0.]])

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        bbox = self._state_to_bbox()
        self.history.append(bbox)
        return bbox

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.; cy = (y1+y2)/2.
        s = (x2-x1)*(y2-y1)
        r = (x2-x1)/(y2-y1) if (y2-y1)>0 else 1.
        z = np.array([[cx],[cy],[s],[r]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def _state_to_bbox(self):
        cx,cy,s,r = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        if s < 0: s = 1.
        w = math.sqrt(abs(s * r))
        h = s / w if w > 0 else 1.
        x1 = cx - w/2.; y1 = cy - h/2.
        x2 = cx + w/2.; y2 = cy + h/2.
        return [x1, y1, x2, y2]

    def get_state(self):
        return self._state_to_bbox()


class Sort:
    """
    SORT: Simple Online and Realtime Tracking.
    Associates detections to existing tracks using IoU.
    """
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        detections: numpy array Nx4 of [x1,y1,x2,y2]
        returns: numpy array Mx5 of [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # Predict from existing trackers
        trks = []
        to_del = []
        for i, t in enumerate(self.trackers):
            pred = t.predict()
            if any(math.isnan(v) for v in pred):
                to_del.append(i)
            else:
                trks.append(pred)
        for i in reversed(to_del):
            self.trackers.pop(i)

        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, trks)

        # Update matched
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        # Collect results
        results = []
        for t in self.trackers:
            if t.time_since_update < 1 and (
                    t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bb = t.get_state()
                results.append([bb[0], bb[1], bb[2], bb[3], t.id])

        return np.array(results) if results else np.empty((0, 5))

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)

        # Greedy matching
        matched = []
        used_det = set()
        used_trk = set()
        flat = [(iou_matrix[d,t], d, t)
                for d in range(len(detections))
                for t in range(len(trackers))]
        flat.sort(reverse=True)
        for val, d, t in flat:
            if val < self.iou_threshold:
                break
            if d not in used_det and t not in used_trk:
                matched.append((d, t))
                used_det.add(d)
                used_trk.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in used_det]
        unmatched_trks = [t for t in range(len(trackers)) if t not in used_trk]
        return matched, unmatched_dets, unmatched_trks


# ── Detector: try YOLOv8 (ultralytics), fall back to YOLOv3 (OpenCV DNN) ────

def try_ultralytics(model_name="yolov8s.pt"):
    """
    Load a YOLOv8 model via ultralytics. Defaults to yolov8s (small), which is
    notably more accurate than yolov8n (nano) at telling visually similar
    objects apart (e.g. phone vs remote vs book vs wallet) while still being
    fast enough for real-time use on CPU. Falls back to yolov8n if yolov8s
    cannot be downloaded (e.g. no internet), and finally gives up gracefully.
    """
    try:
        from ultralytics import YOLO
        try:
            model = YOLO(model_name)
        except Exception:
            # fall back to the nano model bundled/cached if larger one unavailable
            model = YOLO("yolov8n.pt")
            model_name = "yolov8n.pt"
        return model, "yolov8"
    except Exception:
        return None, None


def download_file(url, dest):
    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def load_yolov3(weights_dir="."):
    cfg = os.path.join(weights_dir, "yolov3.cfg")
    weights = os.path.join(weights_dir, "yolov3.weights")
    names = os.path.join(weights_dir, "coco.names")

    if not os.path.exists(cfg):
        download_file(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            cfg)
    if not os.path.exists(names):
        download_file(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            names)
    if not os.path.exists(weights):
        print("  YOLOv3 weights (~236 MB) not found. Downloading...")
        download_file(
            "https://pjreddie.com/media/files/yolov3.weights",
            weights)

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names) as f:
        classes = [l.strip() for l in f.readlines()]
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, out_layers


def detect_yolov3(net, out_layers, classes, frame, conf_thresh=0.5, nms_thresh=0.45):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < conf_thresh:
                continue
            cx,cy,bw,bh = det[0]*w, det[1]*h, det[2]*w, det[3]*h
            x1 = int(cx - bw/2); y1 = int(cy - bh/2)
            boxes.append([x1, y1, int(bw), int(bh)])
            confidences.append(conf)
            class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,bw,bh = boxes[i]
            results.append({
                "box": [x, y, x+bw, y+bh],
                "class": classes[class_ids[i]],
                "conf": confidences[i],
            })
    return results


def detect_yolov8(model, frame, conf_thresh=0.5, iou_thresh=0.45, imgsz=640):
    """
    Run YOLOv8 inference with explicit NMS IoU control and a larger default
    inference resolution. A higher imgsz (e.g. 640 instead of the implicit
    smaller default) substantially improves accuracy on small or handheld
    objects (phones, remotes, cups) which is the most common source of
    misclassification at default settings. agnostic_nms=False keeps NMS
    class-aware so overlapping objects of different classes (e.g. a hand
    holding a phone) are not incorrectly merged into one box.
    """
    results = model(
        frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        agnostic_nms=False,
        verbose=False,
    )[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        detections.append({"box": [x1, y1, x2, y2], "class": label, "conf": conf})
    return detections


# ── Drawing helpers ───────────────────────────────────────────────────────────

COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),
    (255,0,255),(128,0,128),(255,128,0),(0,128,255),(128,255,0),
]

def color_for_id(tid):
    return COLORS[int(tid) % len(COLORS)]

def draw_tracked(frame, tracked, det_map, tracker_ids):
    """
    tracked: Mx5 array [x1,y1,x2,y2,id]
    det_map: dict id -> {"class", "conf"}
    """
    for row in tracked:
        x1,y1,x2,y2 = int(row[0]),int(row[1]),int(row[2]),int(row[3])
        tid = int(row[4])
        col = color_for_id(tid)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        info = det_map.get(tid, {})
        cls = info.get("class", "obj")
        conf = info.get("conf", 0.)
        label = f"ID:{tid} {cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = max(y1-4, th+4)
        cv2.rectangle(frame, (x1, by-th-4), (x1+tw+4, by), col, -1)
        cv2.putText(frame, label, (x1+2, by-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Object Detection and Tracking")
    parser.add_argument("--source", default="0",
                        help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Detection confidence threshold (default 0.45)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold for suppressing duplicate/overlapping boxes (default 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size; larger values improve accuracy on small objects like phones (default 640)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLOv8 model weights to use: yolov8n.pt (fastest), yolov8s.pt (default, more accurate), yolov8m.pt (most accurate, slower)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window (useful for servers)")
    args = parser.parse_args()

    # Open video source
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source '{args.source}'")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Source: {args.source}  Resolution: {width}x{height}  FPS: {fps:.1f}")

    # Setup output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    # Load detector
    print("Loading detector...")
    yolo8_model, backend = try_ultralytics(args.model)
    if backend == "yolov8":
        print(f"  Using YOLOv8 (ultralytics) - model: {args.model}, imgsz: {args.imgsz}, conf: {args.conf}, iou: {args.iou}")
    else:
        print("  ultralytics not available; loading YOLOv3 via OpenCV DNN")
        yolo3_net, coco_classes, yolo3_out_layers = load_yolov3()
        backend = "yolov3"

    # Init SORT tracker
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    print("Tracker: SORT (Kalman filter + IoU association)")
    print("Press Q to quit.\n")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Detection
        if backend == "yolov8":
            detections = detect_yolov8(yolo8_model, frame, args.conf, args.iou, args.imgsz)
        else:
            detections = detect_yolov3(yolo3_net, yolo3_out_layers,
                                       coco_classes, frame, args.conf, args.iou)

        # Convert detections to numpy for SORT
        if detections:
            dets_np = np.array([[d["box"][0],d["box"][1],
                                  d["box"][2],d["box"][3]] for d in detections],
                               dtype=float)
        else:
            dets_np = np.empty((0, 4))

        # Track
        tracked = tracker.update(dets_np)

        # Map track IDs to class/conf (best IoU match)
        det_map = {}
        for row in tracked:
            tid = int(row[4])
            tx1,ty1,tx2,ty2 = row[0],row[1],row[2],row[3]
            best_iou = 0.
            best_det = None
            for d in detections:
                v = iou([tx1,ty1,tx2,ty2], d["box"])
                if v > best_iou:
                    best_iou = v
                    best_det = d
            if best_det:
                det_map[tid] = best_det

        # Draw bounding boxes and labels
        frame = draw_tracked(frame, tracked, det_map, [])

        # Stats overlay
        cv2.putText(frame,
                    f"Frame:{frame_num}  Tracks:{len(tracked)}  Dets:{len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Backend: {backend}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        if writer:
            writer.write(frame)

        if not args.headless:
            cv2.imshow("Object Detection + Tracking (SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit by user.")
                break
        else:
            if frame_num % 30 == 0:
                print(f"Frame {frame_num}: {len(tracked)} tracks, "
                      f"{len(detections)} detections")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
