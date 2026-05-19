# SwingSight AI: Club-Aware Golf Swing Coach

## Project Overview

**SwingSight AI** is a computer vision and deep learning project designed to analyze golf swing mechanics using video-based pose estimation and club-aware feedback logic.

The system first identifies or confirms the golf club being used, then analyzes a golfer’s swing video to extract key body mechanics such as posture, head movement, knee flex, shoulder rotation, hip rotation, balance, and tempo. Based on the detected or selected club type, the system adjusts its feedback so that a driver swing is evaluated differently than an iron or wedge swing.

This project is being developed for a master’s-level AI and Deep Learning course. The goal is to build a realistic, demo-friendly, technically meaningful computer vision system that combines object detection, pose estimation, feature extraction, and interpretable feedback in a polished Streamlit dashboard.

---

## Problem Statement

Golf swing analysis is often expensive, subjective, or dependent on specialized equipment such as launch monitors, professional coaching sessions, or high-speed camera systems. Many amateur golfers record their swings on phones but lack a structured way to interpret what they are seeing.

Most simple golf video tools provide slow-motion playback but do not extract measurable movement patterns. SwingSight AI aims to bridge that gap by using computer vision to turn normal swing video into interpretable swing metrics.

A key challenge in golf analysis is that swing expectations change depending on the club. A driver swing, iron swing, and wedge swing do not have identical setup positions, movement patterns, or tempo expectations. SwingSight AI addresses this by adding a **club-aware analysis layer**, allowing feedback to adjust based on the club category.

---

## Project Objectives

The main objectives of this project are to:

1. Build a computer vision pipeline that can process golf swing videos.
2. Use pose estimation to track golfer body landmarks throughout the swing.
3. Extract meaningful swing mechanics from video frames.
4. Add a pre-swing club identification or confirmation step.
5. Adjust swing feedback based on club category.
6. Display results in a clean, interactive Streamlit dashboard.
7. Produce interpretable, coaching-style recommendations grounded in calculated metrics.
8. Create a realistic MVP that can be demonstrated successfully within a master’s-level course timeline.

---

## Core Concept

The project follows this general workflow:

1. The user uploads an image of the golf club or selects the club manually.
2. The system detects or confirms the club category.
3. The user uploads a golf swing video.
4. The system runs pose estimation on the golfer.
5. Key swing frames and body landmarks are extracted.
6. Swing metrics are calculated.
7. The system applies club-specific scoring logic.
8. Results are shown in a dashboard with metrics, feedback, and annotated visuals.

---

## MVP Scope

The minimum viable product focuses on a reliable and polished workflow rather than overly complex features.

### MVP Features

- Club image upload
- Broad club category detection or manual confirmation
- Supported club categories:
  - Driver / wood-style club
  - Hybrid
  - Iron / wedge-style club
- Manual exact-club confirmation using a dropdown
- Golf swing video upload
- Pose estimation on swing video
- Frame-by-frame landmark extraction
- Basic swing metric calculation
- Club-specific feedback rules
- Streamlit dashboard interface
- Annotated key frames or annotated video output
- Swing scorecard and recommendations

---

## Stretch Goals

The following features may be added if time allows:

- Exact club number recognition using OCR or image classification
- Automatic swing phase detection
- Reference swing comparison against a skilled golfer
- Downloadable PDF or Word swing report
- Real-time webcam club scan
- LLM-generated coaching summary based only on calculated metrics
- Swing consistency tracking across multiple practice sessions
- Support for front-view and down-the-line camera angles
- More advanced temporal deep learning model for swing phase classification

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
            |-- Optional Report Generation
