# CodeAlpha_ObjectDetectionTracking
### Task 4: Real-Time Object Detection and Tracking — CodeAlpha Artificial Intelligence Internship

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF.svg)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time deep learning computer vision system that performs high-speed object detection and persistent multi-object tracking across video streams using **Ultralytics YOLOv8 (Nano)** and the **ByteTrack** multi-object tracking algorithm, augmented with motion trajectory history and a heads-up display (HUD).

---

## 📌 Features

- **State-of-the-Art Detection**: Employs lightweight, real-time YOLOv8 (nano) trained on MS-COCO across 80 common object classes (people, vehicles, animals, appliances, etc.).
- **Robust Multi-Object Tracking**: Uses ByteTrack with Kalman filtering and Hungarian matching to retain persistent IDs across occlusion and rapid movement.
- **Dynamic Motion Trajectories**: Renders continuous path-trail polylines following the center of moving objects.
- **Heads-Up Display (HUD)**: Overlays live metrics including real-time FPS, total active tracked objects, and class distributions.
- **Multiple Input Sources**: Works with real-time webcams (`0`), video files (`.mp4`, `.avi`), or built-in fallback synthetic video generators.
- **Video Recording & Snapshots**: Allows live recording to file (`output_tracked.mp4`) and single-key screenshot saving.

---

## 🛠️ Tech Stack

- **Computer Vision**: [OpenCV (opencv-python)](https://opencv.org/)
- **Detection & Tracking Engine**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **Deep Learning Backend**: [PyTorch](https://pytorch.org/)
- **Language**: Python 3.9+

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/a-b-infotech/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tracking on Webcam
```bash
python tracker.py --source 0
```

### 4. Run Tracking on Video File and Record Output
```bash
python tracker.py --source your_video.mp4 --save --output output_tracked.mp4
```

### Interactive Controls
- `q` : Quit tracker and close display window.
- `s` : Save an instant snapshot of the annotated frame to disk.
- `c` : Clear historical motion trajectory lines.

---

## 📖 Project Structure
```
CodeAlpha_ObjectDetectionTracking/
│
├── tracker.py          # Real-time YOLOv8 detection and ByteTrack tracking engine
├── yolov8n.pt          # Pre-trained YOLOv8 Nano model weights
├── requirements.txt    # Project dependencies
└── README.md           # Documentation & instructions
```

---

## 👨‍💻 Author
- **Name**: Abhi B Sheth
- **Internship**: CodeAlpha Artificial Intelligence Internship
- **GitHub**: [@a-b-infotech](https://github.com/a-b-infotech)
