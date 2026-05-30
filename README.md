# SwingSight AI: Club-Aware Golf Swing Coach

## Project Overview

SwingSight AI analyzes golf swings from local video using computer vision and modular AI components. The dashboard runs as a **local web app** (Flask backend + HTML/CSS/JavaScript frontend) while keeping the same project goals and model stack:

- Club/head detection using YOLOv8 object detection
- Club category classification using a CNN
- Number/loft recognition using OCR or custom image classification
- Pose estimation using YOLOv8-pose
- Video processing using OpenCV
- Metrics/scoring using pandas, NumPy, and scikit-learn
- Report output using PDF/Word export
- Body-part tracking for feet, knees, hips, spine angle, shoulders, elbows, hands, neck, and head

This project is local-first and intentionally scoped for research/demo workflows, not production deployment.

## Local Web Architecture (Recommended)

**Recommended stack: Flask + server-rendered HTML/CSS/JavaScript**

Why Flask is a strong fit for this project:

- Simple local setup and low overhead for single-user workflows
- Easy static/template integration for dashboard UI
- Straightforward REST endpoints for upload, analysis, and report export
- Keeps Python-centric AI pipeline modular and easy to debug

FastAPI is also excellent, but Flask is better here for a lightweight local dashboard foundation where synchronous request flow and simple template/static serving are enough.

## Updated Project Structure

```text
SwingSight-AI/
  assets/
  data/
  models/
  notebooks/
  outputs/
  reports/
  uploads/
  src/
    app.py
    swingsight/
      __init__.py
      club_detection.py
      config.py
      feedback.py
      io_utils.py
      metrics.py
      pipeline.py
      pose_estimation.py
      visualization.py
    webapp/
      __init__.py
      inference/
        __init__.py
        pipeline_components.py
      routes/
        __init__.py
        dashboard.py
      services/
        __init__.py
        analysis_service.py
        report_service.py
      static/
        css/
          dashboard.css
        js/
          dashboard.js
      templates/
        dashboard.html
      utils/
        __init__.py
        storage.py
  tests/
    test_smoke.py
  config.example.yaml
  requirements.txt
  README.md
```

## Backend Endpoints

- `GET /`: Local dashboard page
- `POST /api/upload-video`: Upload swing video
- `POST /api/upload-club-image`: Upload optional club image
- `POST /api/analyze`: Run full swing analysis pipeline
- `GET /api/results/<analysis_id>`: Return saved analysis result JSON
- `GET /uploads/<filename>`: Serve uploaded assets for local preview
- `GET /outputs/<filename>`: Serve processed outputs (annotated video/images)
- `POST /api/reports/<analysis_id>`: Generate PDF or DOCX report
- `GET /reports/<filename>`: Download report file

## Frontend-Backend Communication

The dashboard JavaScript calls backend APIs with `fetch`:

1. Upload files via `multipart/form-data`
2. Receive upload IDs from backend
3. Submit analysis request as JSON (`video_upload_id`, optional `club_image_upload_id`, optional `club_category`)
4. Render returned JSON metrics, score, and feedback in the UI
5. Request PDF/DOCX generation and trigger local file download

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the sample config:

```bash
cp config.example.yaml config.yaml
```

## Run Locally

From project root:

```bash
python app.py
```

Open the local dashboard in your browser:

```text
http://127.0.0.1:8000
```

Do not open `src/webapp/templates/dashboard.html` directly from the file system.
The dashboard is a Flask template and must be served by the local backend.

## Camera-based Guided Capture (New)

This release adds a local camera-guided workflow so you can use your webcam to capture club and swing data directly.

How it works:

1. Open the dashboard in your browser (`http://127.0.0.1:8000`). The app will ask for permission to use the camera.
2. Click the **Record Swing** button to start a guided capture.
  - The app will first ask you to show the butt/end of your club to the camera for club recognition.
  - If the club is confirmed, the app will check that your full body is visible.
  - If your body is visible, the app will automatically record a short clip of your swing and send it to the backend for analysis.
3. When analysis finishes, results are shown in the dashboard and you can download a PDF or Word report.

Notes:
- The current club/body detectors are placeholders. Replace with YOLOv8/CNN/OCR and YOLOv8-pose for production accuracy.
- The guided capture records a short clip automatically (approx 5s); this can be adjusted in `src/webapp/static/js/dashboard.js`.
- The entire workflow is local-first. No credentials, cloud storage, or external services are used by default.

## Notes on Scope

This refactor keeps the project local-first and out of production scope:

- No authentication
- No cloud deployment
- No database persistence layer
- No multi-user/session management

The new web layer is a local orchestration and visualization shell around your existing modular AI pipeline.

## License

This project is intended for academic use. Add a license if you plan to share publicly.
