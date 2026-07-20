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

### CUDA acceleration

This project is configured for the local NVIDIA RTX 4070 SUPER. The requirements
file pins the CUDA 12.8 PyTorch build, so a fresh environment does not silently
install the CPU-only wheel. Verify GPU access after dependency installation:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

The CNN uses cuda:0 and the YOLO paths use device 0 whenever CUDA is available.
The five-way CNN training notebook requires CUDA and stops with a clear error
instead of silently training on CPU.

## Updated Project Structure

```text
SwingSight-AI/
  assets/
  data/
  models/
  notebooks/
  outputs/
    experiments/
    overlays/
  reports/
  uploads/
  src/
    app.py
    swingsight/
      __init__.py
      club_detection.py
      club_recognition.py
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
  notebooks/
    01_golfdb_video_model_training.ipynb
    02_authorized_driver_dataset_builder.ipynb
  data/
    raw/
    processed/
    frames/
    splits/
  models/
    pretrained/
    trained/
  outputs/
    experiments/
    overlays/
  config.example.yaml
  requirements.txt
  README.md
```

## Research Notebook

The notebook [notebooks/01_golfdb_video_model_training.ipynb](notebooks/01_golfdb_video_model_training.ipynb) is a separate research and training workspace for the Kaggle golf swing videos dataset. It is designed to help with preprocessing, frame extraction, pose experiments, weak labeling, and artifact export without changing the scope of the local dashboard.

What it does:

- downloads or reuses the Kaggle dataset locally
- inventories videos and metadata
- extracts frames into `data/frames/`
- creates train/validation/test split files in `data/splits/`
- tests pretrained YOLOv8 pose and detector models
- saves overlays and experiment outputs in `outputs/experiments/` and `outputs/overlays/`
- prepares files that can later feed the backend and dashboard

Recommended Kaggle setup:

```bash
pip install kagglehub opencv-python ultralytics torch torchvision pandas numpy scikit-learn matplotlib tqdm pillow --no-warn-script-location
```

If you use the Kaggle API directly, place `kaggle.json` in `~/.kaggle/` before downloading.

### Authorized driver-head dataset notebook

[notebooks/02_authorized_driver_dataset_builder.ipynb](notebooks/02_authorized_driver_dataset_builder.ipynb) prepares driver-head images for the `wood_type` classifier. USGA's terms prohibit automated scraping of its services, so the notebook deliberately does not crawl or enumerate the Equipment Database. It accepts only a direct-image manifest supplied under USGA's written authorization or an official export, enforces the approved manufacturer list, records provenance, validates images, removes exact duplicates, and makes model-grouped train/validation/test splits.

## Backend Endpoints

- `GET /`: Local dashboard page
- `POST /api/upload-video`: Upload swing video
- `POST /api/upload-club-image`: Upload optional club image
- `POST /api/analyze`: Run full swing analysis pipeline
- `POST /api/club-detect`: Detect and classify club from a camera frame
- `POST /api/body-check`: Check body visibility from a camera frame
- `POST /api/record-swing`: Upload recorded swing video
- `POST /api/analyze-swing`: Analyze a recorded swing video
- `GET /api/results/<analysis_id>`: Return saved analysis result JSON
- `GET /uploads/<filename>`: Serve uploaded assets for local preview
- `GET /outputs/<filename>`: Serve processed outputs (annotated video/images)
- `POST /api/reports/<analysis_id>`: Generate PDF or DOCX report
- `GET /reports/<filename>`: Download report file

## Beginner-Friendly Local Workflow

The landing page gives two clear choices:

1. **Upload a Swing Video**
  - Pick a video file from your computer
  - The app uploads the video to the backend
  - The backend runs the full SwingSight AI analysis
  - Results show a simple swing summary with coach-style takeaways

2. **Record a Swing**
  - The app requests camera access
  - The app guides you through club recognition, body visibility, and swing recording
  - Short prompts keep things simple:
    - “Show the end of your club.”
    - “Club confirmed: 7 Iron.”
    - “Step back so your full body is visible.”
    - “Ready. Swing away.”
    - “Analyzing your swing...”
  - The backend analyzes the recorded swing and shows the same results

The main results view is intentionally simple (score, confirmed club, biggest takeaways, and one focus item). An **Advanced Details** section is available if you want to explore deeper metrics and model outputs.

## Frontend-Backend Communication

The dashboard JavaScript calls backend APIs with `fetch`:

1. Upload files via `multipart/form-data`
2. Receive upload IDs from backend
3. Submit analysis request as JSON (`video_upload_id`)
4. Render returned JSON metrics, score, and feedback in the UI
5. Request PDF/DOCX generation and trigger local file download

## Notebook Output to App Connection

The notebook supports the model-building workflow, while the app stays beginner-friendly.

- save trained weights to `models/trained/`
- keep base or downloaded weights in `models/pretrained/`
- save processed metadata to `data/processed/`
- save extracted frames to `data/frames/`
- save train/validation/test split files to `data/splits/`
- save overlays and experiment outputs to `outputs/experiments/` and `outputs/overlays/`
- load the final inference helpers from the Python backend modules
- keep the dashboard focused on coach-style feedback and avoid exposing technical model details by default

## Backend Response Format (Simple + Advanced)

The backend returns a summary payload for the main UI and a separate `advanced` payload for deeper inspection:

- `summary`: overall score, confirmed club, top takeaways, and focus tip
- `advanced`: metrics, pose frame count, body tracking data, model outputs, and rendered assets

## Club Recognition Decision Tree

The club recognition pipeline is staged and local-first. It crops the club head with YOLOv8 when detector weights are available, then uses three checkpoint-backed CNN processes.

1. **Broad-category CNN** classifies the head as `iron` or `wood`.
2. **Iron-number CNN** runs only for an iron. It sees the lower-center marking crop and returns one of `1` through `9`, producing labels such as `7 Iron`.
3. **Wood-type CNN** runs only for a wood and returns `Driver`, `Wood`, or `Hybrid`.

The final confidence weights the detector (15%) and the two CNN decisions (42.5% each). A missing, mismatched, or low-confidence model never falls back to a shape or size guess: the response is `unavailable` or `uncertain` with the precise reason. If a YOLO club-head model is not installed, the CNNs may classify the submitted frame directly; train with the same framing if you intend to use this fallback.

## Club Recognition Module Structure

The staged pipeline is implemented in [src/swingsight/club_recognition.py](src/swingsight/club_recognition.py), with the model architecture, checkpoint validation, and inference preprocessing in [src/swingsight/club_cnn.py](src/swingsight/club_cnn.py):

- `classify_broad_category()` -> iron vs wood CNN
- `classify_iron_number()` -> 1-9 CNN on the marking crop
- `classify_wood_type()` -> Driver/Wood/Hybrid CNN
- `recognize_club_from_frame()` -> orchestrates the decision tree

### Training the Three Club CNNs

No labelled club-image data or trained club CNN weights are committed to this repository. Create the following data layout, keeping the camera framing consistent with live capture. Store club-head crops for `broad_category` and `wood_type`; for `iron_number`, use a close-up where the stamped number is readable. Depending on the club design, this can be the face, sole, or rear cavity—not the grip-end butt cap.

```text
data/club_cnn/
  broad_category/
    train/{iron,wood}/
    val/{iron,wood}/
  iron_number/
    train/{1,2,3,4,5,6,7,8,9}/
    val/{1,2,3,4,5,6,7,8,9}/
  wood_type/
    train/{driver,wood,hybrid}/
    val/{driver,wood,hybrid}/
```

Train one model per stage, then keep the output names that are already referenced by `config.example.yaml`:

```bash
python scripts/train_club_cnn.py --task broad_category --data-dir data/club_cnn/broad_category --output models/trained/club_broad_cnn.pt
python scripts/train_club_cnn.py --task iron_number --data-dir data/club_cnn/iron_number --output models/trained/club_iron_number_cnn.pt
python scripts/train_club_cnn.py --task wood_type --data-dir data/club_cnn/wood_type --output models/trained/club_wood_type_cnn.pt
```

The checkpoints embed their task, class order, preprocessing values, and validation accuracy. Inference rejects a checkpoint assigned to the wrong stage, so an iron-number model cannot accidentally be used as a wood-type model.

## Club Recognition API Response (Example)

```json
{
  "upload_id": "2c5b0f",
  "file_name": "frame.png",
  "result": {
    "status": "confirmed",
    "detected_category": "Wood",
    "predicted_club": "Driver",
    "confidence": 0.72,
    "reasoning": "broad_category CNN predicted wood; wood_type CNN predicted driver",
    "bbox": [120, 80, 420, 320],
    "ocr": null,
    "sources": {
      "detector": "yolov8",
      "broad_classifier": "cnn",
      "wood_type_classifier": "cnn"
    }
  }
}
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also place local overrides in a `.env` file at the repository root. The app reads `SWINGSIGHT_HOST`, `SWINGSIGHT_PORT`, and `SWINGSIGHT_DEBUG` if present.

Copy the sample config:

```bash
cp config.example.yaml config.yaml
```

## Run Locally

From project root:

```bash
python run.py
```

Or use the platform launcher for your operating system:

```bash
./start.sh
```

On Windows:

```bat
start.bat
```

Open the local dashboard in your browser:

```text
http://localhost:8000
```

Do not open `src/webapp/templates/dashboard.html` directly from the file system.
The dashboard is a Flask template and must be served by the local backend.

## Camera-based Guided Capture

This release adds a local camera-guided workflow so you can use your webcam to capture club and swing data directly.

How it works:

1. Open the dashboard in your browser (`http://127.0.0.1:8000`). The app will ask for permission to use the camera.
2. Click **Record Swing** to start the guided capture.
  - Show the butt/end of your club to the camera for club recognition.
  - Once confirmed, the UI displays: “Club confirmed as: [club]. Swing away.”
  - Step back for the body visibility check.
  - The app records a short swing clip and sends it to the backend for analysis.
3. When analysis finishes, results are shown in the dashboard with report download options.

Notes:
- The current club/body detectors are placeholders. Replace with YOLOv8/CNN/OCR and YOLOv8-pose for production accuracy.
- The guided capture records a short clip automatically (approx 5s); this can be adjusted in `src/webapp/static/js/dashboard.js`.
- The entire workflow is local-first. No credentials, cloud storage, or external services are used by default.

See [PLATFORM_COMPATIBILITY_CHECKLIST.md](PLATFORM_COMPATIBILITY_CHECKLIST.md) for the Mac and Windows validation checklist.

Optional OCR dependencies (not required unless you want club marking detection):

- `easyocr` (recommended for stamped markings)
- `pytesseract` (fallback)

## Notes on Scope

This refactor keeps the project local-first and out of production scope:

- No authentication
- No cloud deployment
- No database persistence layer
- No multi-user/session management

The new web layer is a local orchestration and visualization shell around your existing modular AI pipeline.

## License

This project is intended for academic use. Add a license if you plan to share publicly.
