# SwingSight AI: Club-Aware Golf Swing Coach

## Project Overview

SwingSight AI is a computer vision and deep learning project that analyzes golf swing mechanics using video-based pose estimation and club-aware feedback logic. The system first identifies or confirms the golf club category from an image, then processes a swing video to extract key body mechanics such as posture, head movement, knee flex, shoulder turn, hip turn, balance, and a basic tempo estimate. Feedback is adjusted based on club category so that a driver swing is evaluated differently than an iron or wedge swing.

This project is scoped for a master’s-level AI and Deep Learning course. The MVP focuses on a polished, demo-friendly system with clear metrics and interpretable feedback in a Streamlit dashboard.

---

## Problem Statement

Golf swing analysis is often expensive, subjective, or dependent on specialized equipment. Many golfers record swings with phones but lack tools that turn video into measurable mechanics. SwingSight AI bridges that gap by using computer vision to extract swing metrics from ordinary video and by applying club-aware expectations.

---

## Objectives

1. Build a video processing pipeline for golf swing analysis.
2. Use pose estimation to track golfer body landmarks.
3. Extract meaningful swing metrics from video frames.
4. Add a pre-swing club identification or confirmation step.
5. Generate club-aware feedback and scoring.
6. Present results in a clean Streamlit dashboard.

---

## MVP Scope

### MVP Features

- Club image upload
- Broad club category detection or manual confirmation
- Supported club categories:
  - Driver / wood-style club
  - Hybrid
  - Iron / wedge-style club
- Manual exact-club confirmation via dropdown
- Swing video upload
- Pose estimation on swing video
- Key swing metric calculation
- Club-specific feedback rules
- Streamlit dashboard
- Annotated key frames or video
- Swing scorecard and recommendations

---

## Stretch Goals

- Exact club number recognition using OCR or classification
- Automatic swing phase detection
- Reference swing comparison against a skilled golfer
- Downloadable PDF or Word swing report
- Real-time webcam club scan
- LLM-generated coaching summary grounded only in calculated metrics

---

## System Architecture

```text
User Input
   |
   |-- Club Image Upload / Manual Club Selection
   |        |
   |        |-- Club Detection Model
   |        |-- Club Category Prediction
   |        |-- Manual Confirmation Fallback
   |
   |-- Swing Video Upload
            |
            |-- Video Frame Extraction
            |-- Pose Estimation
            |-- Landmark Tracking
            |-- Swing Metric Calculation
            |-- Club-Specific Feedback Logic
            |-- Dashboard Visualization
```

---

## Data Collection Plan

- Club images: 50 to 150 images per category (driver/wood, hybrid, iron/wedge). Use personal photos or open-license datasets.
- Swing videos: 30 to 60 videos of varied skill levels, ideally from two camera angles (down-the-line and face-on).
- Annotation: Simple labels for club category and swing angle; optional manual tags for phase boundaries to evaluate tempo.
- Storage: Keep raw data out of GitHub. Use local folders or cloud storage (Drive, OneDrive, S3) and document paths in config.

---

## Model Approach

- Club detection: Start with a lightweight image classifier or YOLOv8 model for broad club categories. A manual override is always available.
- Pose estimation: Use MediaPipe Pose for fast, reliable landmarks or YOLOv8-pose for higher quality if GPU is available.
- Metrics: Compute geometric proxies (angles and distances) from landmarks, then normalize by torso or shoulder width.
- Feedback: Rule-based scoring logic keyed by club category to keep results interpretable and realistic for the MVP.

---

## Dashboard Workflow

1. Upload club image or select club category manually.
2. Confirm the detected category with a dropdown.
3. Upload swing video.
4. Run analysis and display:
   - Detected club type and confidence
   - Annotated key frames or video
   - Metrics scorecard
   - Club-specific recommendations

---

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the sample config:

```bash
copy config.example.yaml config.yaml
```

---

## Usage

```bash
streamlit run src/app.py
```

---

## Project Structure

```text
SwingSight-AI/
  assets/
  data/
  models/
  outputs/
  reports/
  notebooks/
  src/
    app.py
    swingsight/
      __init__.py
      config.py
      pipeline.py
      club_detection.py
      pose_estimation.py
      metrics.py
      feedback.py
      visualization.py
      io_utils.py
  tests/
  config.example.yaml
  requirements.txt
  README.md
  .gitignore
```

---

## Evaluation Plan

- Club detection: report accuracy on a held-out image set.
- Pose tracking: visually inspect key frames and compare landmark stability.
- Metric stability: compute variance across multiple swings of the same golfer.
- Feedback sanity checks: compare outputs to coaching expectations for driver vs iron.

---

## Risks and Limitations

- Pose estimation can fail in low light or with occlusions.
- Camera angle changes can affect metric consistency.
- Club detection is broad-category only in the MVP.
- Tempo estimates are approximate without explicit swing phase labels.

---

## Future Improvements

- Add swing phase detection to improve tempo and sequencing feedback.
- Expand club classification to exact club number.
- Add reference swing comparison and historical tracking.
- Generate a downloadable report.

---

## License

This project is intended for academic use. Add a license if you plan to share publicly.
