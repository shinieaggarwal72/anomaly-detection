import argparse
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    history: Deque[Tuple[float, float, float]]
    is_running: bool = False
    is_loitering: bool = False


def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def draw_label(img, text, x, y):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - baseline - 4), (x + w + 4, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def pixel_speed(history: Deque[Tuple[float, float, float]]):
    if len(history) < 2:
        return 0.0
    (t1, x1, y1) = history[-2]
    (t2, x2, y2) = history[-1]
    dt = max(1e-6, t2 - t1)
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return dist / dt


def loitering_score(history: Deque[Tuple[float, float, float]]):
    if len(history) < 2:
        return 0.0, 0.0
    times = [t for (t, _, _) in history]
    xs = [x for (_, x, _) in history]
    ys = [y for (_, _, y) in history]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r = float(max((((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) for x, y in zip(xs, ys)))
    duration = times[-1] - times[0]
    return r, duration


def process_video(
    source: str,
    output: str,
    model_name: str = "yolov8n.pt",
    tracker_cfg: str = "bytetrack.yaml",
    run_thresh_px_per_s: float = 250.0,
    loiter_time_s: float = 20.0,
    loiter_radius_px: float = 40.0,
    window_s: float = 25.0,
    show_window: bool = True,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_parent_dir(output)
    if not output.endswith(".mp4"):
        output = os.path.splitext(output)[0] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))


    model = YOLO(model_name)

    track_states: Dict[int, TrackState] = defaultdict(lambda: TrackState(history=deque()))

    frame_idx = 0
    results_generator = model.track(
        source=source,
        tracker=tracker_cfg,
        stream=True,
        persist=True,
        classes=[0], 
        verbose=False,
    )

    for result in results_generator:
        frame = result.orig_img.copy()
        t = frame_idx / max(1e-6, fps)

        
        active_ids = set()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes))
            ids = result.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else np.array([-1] * len(boxes))

            for (x1, y1, x2, y2), conf, tid in zip(boxes, confs, ids):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                state = track_states[tid]

                state.history.append((t, cx, cy))
                while state.history and (t - state.history[0][0]) > window_s:
                    state.history.popleft()

                speed = pixel_speed(state.history)
                radius, duration = loitering_score(state.history)

                state.is_running = speed > run_thresh_px_per_s
                state.is_loitering = (duration >= loiter_time_s) and (radius <= loiter_radius_px)

                
                active_ids.add(tid)

                
                color = (0, 255, 0)
                if state.is_running:
                    color = (0, 0, 255)
                elif state.is_loitering:
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid} {conf:.2f}"
                if state.is_running:
                    label += " | RUNNING"
                elif state.is_loitering:
                    label += " | LOITERING"
                draw_label(frame, label, x1, y1)

                if state.is_loitering:
                    cv2.circle(frame, (int(cx), int(cy)), int(loiter_radius_px), (0, 165, 255), 1, cv2.LINE_AA)

        
        total_persons = len(active_ids)
        running_count = sum(1 for tid in active_ids if track_states[tid].is_running)
        loiter_count = sum(1 for tid in active_ids if track_states[tid].is_loitering)

        hud = f"Persons: {total_persons}  Running: {running_count}  Loitering: {loiter_count}"
        draw_label(frame, hud, 10, 30)

        out.write(frame)
        if show_window:
            cv2.imshow("Anomaly Detection (Running/Loitering)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 person tracking with running and loitering anomaly detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input .avi video")
    parser.add_argument("--output", type=str, default="anomaly_output.avi", help="Path to save annotated video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config for ultralytics (e.g., bytetrack.yaml)")
    parser.add_argument("--run_thresh", type=float, default=200.0, help="Running threshold in pixels/sec")
    parser.add_argument("--loiter_time", type=float, default=8.0, help="Seconds to consider as loitering")
    parser.add_argument("--loiter_radius", type=float, default=50.0, help="Radius in pixels for loitering")
    parser.add_argument("--window", type=float, default=25.0, help="Seconds of history to track per ID")
    parser.add_argument("--no_show", action="store_true", help="Do not display live window")
    args = parser.parse_args()

    process_video(
        source=args.video,
        output=args.output,
        model_name=args.model,
        tracker_cfg=args.tracker,
        run_thresh_px_per_s=args.run_thresh,
        loiter_time_s=args.loiter_time,
        loiter_radius_px=args.loiter_radius,
        window_s=args.window,
        show_window=not args.no_show,
    )
