# OOP_Project
Real-Time Object Detection with Event-Based Capture

# VisionGuard: Real-Time Object Detection & Event Trigger System

## Overview

VisionGuard is a real-time computer vision system designed to detect objects and trigger actions based on meaningful events.

Instead of continuously processing and storing redundant detections, the system applies an **event-driven approach**, capturing outputs only when new target objects appear in the scene.

This project demonstrates how to move from simple object detection to a **modular, scalable, and intelligent vision pipeline**.

---

## Key Features

*  Real-time object detection using YOLOv8 (Ultralytics)
*  Modular OOP architecture (clean and extensible)
*  Event-based triggering (no redundant outputs)
*  Automatic image capture on new detections
*  Class filtering (e.g., ignoring "person")
*  Side-by-side visualization (raw vs processed frame)
*  Jupyter Notebook support for rapid experimentation

---

##  Project Structure

```
vision_project/
│
├── src/
│   ├── core/
│   │   ├── detector.py       # Model inference (YOLO)
│   │   ├── processor.py      # Business logic & filtering
│   │   ├── visualizer.py     # Drawing bounding boxes
│   │   └── pipeline.py       # End-to-end orchestration
│   │
│   └── utils/
│       └── helpers.py        # Utility functions (optional)
│
├── notebooks/
│   └── demo.ipynb            # Main demo (Jupyter)
│
├── outputs/                  # Saved detection images
├── requirements.txt
└── README.md
```

---

##  Architecture

The system is designed with **separation of responsibilities**:

* **Detector**
  Handles model loading and inference using YOLO.

* **Processor**
  Applies filtering logic, ignores irrelevant classes, and selects target objects.

* **Visualizer**
  Draws bounding boxes and labels on frames.

* **Pipeline**
  Connects all components into a unified processing flow.

This structure allows easy extension to:

* new models
* new detection rules
* new output systems (API, alerts, logging)

---

##  Use Case Example

The current implementation:

* Detects objects such as **cell phones**
* Ignores irrelevant detections (e.g., "person")
* Saves an image **only when a new object appears**

Example output:

```
 New detection: ['cell phone'] → outputs/detect_1700000000.jpg
```

---

##  How It Works

1. Capture frame from webcam
2. Run object detection (YOLO)
3. Filter detections based on rules
4. Compare with previous state
5. Trigger event if new object is detected
6. Save image + display result

---

##  Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* Matplotlib (for notebook visualization)

---

##  Getting Started

### 1. Install dependencies

```bash
pip install ultralytics opencv-python matplotlib
```

### 2. Run the project

Open the notebook:

```bash
notebooks/demo.ipynb
```

Run all cells to start the real-time detection system.

---

##  Controls

* Stop execution using Jupyter's stop button (⏹️)
* The system will safely release the camera after interruption

---

##  Outputs

Detected events are saved automatically:

```
outputs/
├── detect_1700000000.jpg
├── detect_1700000005.jpg
```

Images are saved **only when a new object is detected**, avoiding redundancy.

---

##  Notes & Limitations

* The default YOLO model (COCO dataset) may not accurately detect:

  * Smartwatches (often classified as "cell phone")
* Performance in Jupyter is not real-time due to rendering constraints
* For full performance, consider running as a `.py` script

---

## Future Improvements

* 🔄 Object tracking (DeepSORT)
* 🌐 API deployment (FastAPI)
* 📲 Real-time alerts (Telegram / Webhooks)
* 🧠 Custom-trained models (domain-specific detection)
* ⚡ Edge deployment (Jetson Nano / Raspberry Pi)
* 📊 Detection logging (JSON + analytics)

---

##  Key Insight

This project goes beyond basic object detection by implementing:

> **Event-driven vision systems**

A critical concept for real-world applications such as:

* Surveillance systems
* Industrial monitoring
* Robotics perception
* Smart automation pipelines

---

## 👨‍💻Author

Developed as part of practical exploration in Computer Vision, focusing on:

* modular system design
* real-time processing
* intelligent event handling

---

