import os
import sys
import time
import argparse
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def create_synthetic_demo_video(filename="demo_input.mp4", duration_sec=5, fps=24):
    """Creates a simple synthetic video with moving shapes if no camera/video exists."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    total_frames = duration_sec * fps

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = int(20 + 20 * (i / total_frames))
        frame[:, :, 1] = int(25)
        frame[:, :, 2] = int(35)

        # Draw moving target 1
        cx1 = int(100 + (width - 200) * (i / total_frames))
        cy1 = int(240 + 60 * np.sin(i / 10))
        cv2.circle(frame, (cx1, cy1), 35, (0, 255, 128), -1)
        cv2.putText(frame, "Target 1", (cx1 - 30, cy1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

        # Draw moving target 2
        cx2 = int(width - 100 - (width - 200) * (i / total_frames))
        cy2 = int(180 + 40 * np.cos(i / 8))
        cv2.rectangle(frame, (cx2 - 30, cy2 - 30), (cx2 + 30, cy2 + 30), (255, 100, 50), -1)
        cv2.putText(frame, "Target 2", (cx2 - 30, cy2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 50), 2)

        out.write(frame)

    out.release()
    return filename

def draw_hud(frame, fps, active_tracks, class_counts):
    """Draws a professional HUD display on top of the frame."""
    h, w = frame.shape[:2]
    # Header overlay banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title & Stats
    cv2.putText(frame, "CodeAlpha AI | Real-Time YOLOv8 Object Tracking", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
    
    status_text = f"FPS: {fps:.1f} | Active Tracks: {active_tracks} | Press 'q' to exit | 's' to save"
    cv2.putText(frame, status_text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

def run_object_tracker(source="0", model_path="yolov8n.pt", save_output=False, output_path="output_tracked.mp4"):
    """
    Real-Time Object Detection and Tracking using YOLOv8 & ByteTrack.
    """
    # Check model file
    if not os.path.exists(model_path):
        # Look in parent directory if needed
        parent_model = os.path.join("..", model_path)
        if os.path.exists(parent_model):
            model_path = parent_model

    print(f"Loading YOLO model from: {model_path}...")
    model = YOLO(model_path)

    # Resolve Video Source
    is_webcam = source.isdigit()
    video_src = int(source) if is_webcam else source

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Warning: Could not open source '{source}'.")
        if is_webcam:
            print("Webcam not detected. Creating synthetic demo video to verify pipeline...")
            demo_video = create_synthetic_demo_video()
            cap = cv2.VideoCapture(demo_video)
        else:
            print("Exiting tracker.")
            return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"Recording output stream to '{output_path}'...")

    # Trajectory history for movement trails
    track_history = defaultdict(lambda: [])
    prev_time = time.time()
    fps = 0.0

    print("=" * 60)
    print("Tracker started successfully!")
    print("Keyboard shortcuts:")
    print("  [q] : Quit tracking")
    print("  [s] : Save current frame snapshot")
    print("  [c] : Clear trajectory trails")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or feed interrupted.")
                break

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
            prev_time = curr_time

            # Run YOLO Tracking (ByteTrack / BoT-SORT)
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            class_counts = defaultdict(int)
            active_tracks = 0

            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                names = model.names

                active_tracks = len(track_ids)

                # Draw bounding boxes and tracks
                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    x, y, w, h = box
                    class_name = names[cls_id]
                    class_counts[class_name] += 1

                    # Center coordinate
                    cx, cy = int(x), int(y)
                    track = track_history[track_id]
                    track.append((cx, cy))
                    if len(track) > 30:  # Retain last 30 positions for trail
                        track.pop(0)

                    # Draw smooth motion trajectory trail
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 220, 255), thickness=2)

                    # Bounding box corners
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    color = (0, 200, 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 9, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw HUD Overlays
            draw_hud(frame, fps, active_tracks, class_counts)

            if video_writer is not None:
                video_writer.write(frame)

            # Display window
            cv2.imshow("CodeAlpha AI - Object Detection & Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap_name = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_name, frame)
                print(f"Saved snapshot to {snap_name}")
            elif key == ord('c'):
                track_history.clear()
                print("Cleared trajectory trails.")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Tracker stopped cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeAlpha AI Object Detection & Tracking")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--save", action="store_true", help="Record annotated video stream to file")
    parser.add_argument("--output", type=str, default="output_tracked.mp4", help="Saved video path")
    args = parser.parse_args()

    run_object_tracker(
        source=args.source,
        model_path=args.model,
        save_output=args.save,
        output_path=args.output
    )
