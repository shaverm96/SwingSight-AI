# SwingSight AI

SwingSight AI is a local-first golf swing coaching application. It accepts a short swing video from a camera roll or a live browser recording, analyzes body movement from the video, identifies the club when possible, and turns the result into a practical review with motion overlays, KPI rankings, coaching guidance, and a downloadable PDF report.

The product is designed for a practice session: upload or record a swing, let the local analysis run, then review one clear priority and take a coach-style cue to the next ball.

> SwingSight provides video-based practice guidance, not medical advice, an injury assessment, a launch-monitor replacement, or a substitute for instruction from a qualified golf professional.

## Contents

- [What SwingSight does](#what-swingsight-does)
- [How a swing moves through the system](#how-a-swing-moves-through-the-system)
- [Using the app](#using-the-app)
- [Review and PDF report](#review-and-pdf-report)
- [Club selection and recognition](#club-selection-and-recognition)
- [Movement analysis and KPIs](#movement-analysis-and-kpis)
- [Optional Gemini coaching](#optional-gemini-coaching)
- [Install and run](#install-and-run)
- [Configuration](#configuration)
- [Local API](#local-api)
- [Storage, privacy, and security](#storage-privacy-and-security)
- [Models, data, and training](#models-data-and-training)
- [Development and testing](#development-and-testing)
- [Troubleshooting](#troubleshooting)

## What SwingSight does

### Landing page and capture choices

The local web app opens to a minimal SwingSight landing page, then offers two capture paths farther down the page:

1. **Upload a swing**
   - Accepts a browser-supported video file (`video/*`). MP4, MOV, and WebM are the intended common formats.
   - Requires the golfer to select a club family before choosing a video.
   - For an Iron or Wedge, the UI also asks for the exact club marking or loft.

2. **Record a swing**
   - Opens the browser camera through `getUserMedia`.
   - Runs a short countdown, captures a frame for club recognition, checks that the golfer's body is visible, then records a 5.5-second WebM clip.
   - Uses the detected club when recognition is confident enough.

Both paths show a staged in-browser progress sheet while the video is uploaded, analyzed, and prepared for review.

### Analysis and coaching

For a readable swing video, SwingSight can produce:

- Original-video playback and a selectable pose-overlay video.
- Raw and smoothed overlay variants, plus diagnostic assets when the local pipeline creates them.
- A Swing Score, score label, and five player-facing KPI cards.
- A detected or user-selected club.
- A single next-best coaching focus.
- Strengths, improvements, practice cues, coaching observations, and optional drills.
- Advanced metrics, tracking metadata, warnings, and model outputs for local inspection.

If pose tracking, an overlay encoder, or club recognition is unavailable, the result records the limitation and continues where a useful fallback is possible. A missing or weak reading should be treated as unavailable data, not as a statement about swing quality.

### Branded PDF export

The completed review exposes one export action: **Download PDF Report**. SwingSight does not generate or offer Word documents.

The export is a two-page coaching report with:

- SwingSight branding, analysis date, detected club, and a short support ID in the footer.
- A score card with a consistent score status.
- Coach's Take, genuine strengths, next focus, practice cue, and numbered priorities.
- Key movement scores and readable, rounded movement measurements.
- Safe handling for `None`, `NaN`, infinity, suspicious zero values, duplicate values, and raw/internal metric names.
- Page numbers, a practice-guidance disclaimer, and a PDF filename based on the club, such as `SwingSight_8-Iron_Swing_Report.pdf`.

## How a swing moves through the system

```text
Browser upload or live recording
        |
        +-- optional exact club selection (upload)
        |       or club scan + body visibility check (recording)
        v
Flask upload routes save local media
        v
ModelManager validates the video and runs local pose analysis
        |
        +-- YOLO pose inference when the local runtime/model is available
        +-- fallback-frame extraction and local warnings when it is not
        +-- raw and smoothed overlay generation
        +-- club recognition from a frame or submitted club image
        v
Movement metrics + CoachingEngine
        |
        +-- local score, strengths, improvements, focus, observations
        +-- optional Gemini coaching from structured local measurements only
        v
Persisted JSON result in outputs/
        v
Browser review page + optional two-page PDF report in reports/
```

The application is intentionally local-first. The Flask server, files, model inference, overlays, local score, and baseline coaching run on the computer hosting the app. Gemini is the only optional remote service, and it receives structured measurement evidence rather than the source video or raw frames.

## Using the app

1. Start SwingSight and visit `http://127.0.0.1:8000`.
2. Select **Upload a swing** or **Record a swing**.
3. For uploads, choose the club family. If it is an Iron or Wedge, choose the exact number, letter, or loft shown on the club.
4. For live capture, hold the club face or sole clearly in the camera during the countdown. After club confirmation, step back so the full body is visible.
5. Record or choose one short swing video.
6. Wait for the review page to open.
7. Compare the original video with the overlay, read the coaching priority, inspect KPI rankings and practice guidance, then download the PDF report if needed.

### Recording recommendations

For the most useful review:

- Use one swing per clip.
- Keep the golfer, hands, club, and both feet in frame from setup through finish.
- Record from a stable face-on or down-the-line/side view; keep that angle consistent when comparing swings.
- Put the camera roughly waist or hip height and avoid zooming or hand-held movement.
- Use good light and enough contrast between the golfer and background.
- Show a clean, centered club face or sole when club identification matters.
- Keep the clip short. The example configuration uses a 12-second maximum-video target, and shorter clips are faster to inspect and process.

## Review and PDF report

### Review page

The review page contains:

- **KPI rankings:** five visual score cards for the core movement measures.
- **Swing score:** an overall 0-100 score when the available local evidence supports one.
- **Original / Overlay controls:** switches between the uploaded video and the preferred smoothed overlay when it is available.
- **Coach's priority:** the most useful next movement to practice.
- **What I'm seeing:** up to three coach observations.
- **Keep doing / Work on this / Practice cues:** short lists sourced from local or Gemini coaching.
- **Practice plan:** up to three drill cards when drill data is available.
- **Advanced details:** local diagnostic JSON for metrics, tracking, model outputs, and detailed coaching. This section is for development and troubleshooting rather than a player-facing scorecard.

The review page also includes **Record new swing**, which opens the capture workflow directly with `/?record=1`.

### PDF report details

The PDF generator in `src/webapp/services/report_service.py` creates exactly two Letter-size pages under normal conditions.

**Page 1 - Swing Analysis Report**

- SwingSight wordmark, date, club, and AI-powered coaching label.
- Swing Score, score status, detected club, and primary focus.
- Coach's Take.
- Up to three evidence-supported strengths. If no reliable strength is available, the report uses the clearly labeled “Building Your Foundation” fallback rather than inventing a positive observation.
- A visually prominent next-focus card with explanation and practice cue.
- Up to three improvement-priority cards. Each card contains what was observed, why it matters, and a simple swing thought.

**Page 2 - Swing Details**

- A plain-language measurement disclaimer.
- Available normalized scores for Kinematic Sequence, Lateral Weight Shift, Spine Maintenance, and X-Factor.
- A compact Measured Movement table for useful measurements such as spine angle, shoulder rotation, head/hip/knee movement, foot stability, hand path, and lateral-shift ratio.
- Rounded values with units: degrees to one decimal place, pixels to whole numbers, ratios to two decimal places, and scores as whole numbers out of 100.

The report deliberately hides invalid, missing, non-finite, internal-only, and suspicious data instead of exposing Python values or debugging output. The full analysis UUID is not printed in the report body; only a short support ID may appear in the footer.

## Club selection and recognition

SwingSight supports five broad club families:

| Family | Upload selection | Live recognition behavior |
| --- | --- | --- |
| Driver | Driver | Five-way classifier result |
| Wood | Fairway Wood | Five-way classifier result |
| Hybrid | Hybrid | Five-way classifier result |
| Iron | Iron, then `1-9`, `P`, `G`, `A`, or `S` | Five-way classifier, then OCR marking read where possible |
| Wedge | Wedge, then `46`, `48`, `50`, `52`, `54`, `56`, `58`, `60`, `62`, or `64` | Five-way classifier, then OCR marking or loft read where possible |

### Uploaded swings

The upload flow deliberately uses the club chosen in the UI. This avoids making club type a gating condition for the main video analysis and allows a golfer to choose an exact club such as `8 Iron` or `60 Wedge` even if no separate club photo is available.

### Live club scanning

For live recording, SwingSight captures a still frame before recording and uses the following staged workflow:

```text
Hand-based crop when a hand is detected
        v
Adaptive contrast preprocessing
        v
Five-way CNN: Driver | Wood | Hybrid | Iron | Wedge
        v
Only for Iron/Wedge: full-image RapidOCR marking read
        v
Confirmed exact club, broad family, or a clear retry message
```

The hand crop is a convenience heuristic, not a requirement. If MediaPipe's hand landmark model is missing, a hand is not found, or a crop fails, the classifier falls back to the full frame instead of stopping analysis.

### Exact Iron and Wedge markings

RapidOCR runs only after the five-way classifier identifies an Iron or Wedge. It reads the full working image, applies optional contrast enhancement, and only attempts sideways orientations after a normal-orientation pass fails to produce a valid marking.

The normalization layer accepts golf-specific values rather than trusting arbitrary OCR text:

- Iron numbers and common letter markings such as `P/PW`, `G/GW`, `A/AW`, and `S/SW` are treated as Iron designations when valid.
- Numeric markings above 10 are treated as Wedge designations.
- Supported wedge lofts are normalized into a player-facing wedge label.
- Common OCR swaps are corrected only when the corrected result is valid for a golf club.

If OCR is unavailable, below its configured confidence threshold, or cannot read a valid marking, SwingSight keeps the broad Iron or Wedge result. It does not guess an exact club. The response includes `club_type`, `club_number`, `exact_club`, confidence, source, and OCR metadata for local troubleshooting.

### Recognition prerequisites and limitations

- The preferred five-way checkpoint is `models/trained/club_type_5way.pt`.
- `hand_landmarker.task` supports the optional hand-centered crop.
- RapidOCR and ONNX Runtime are installed through `requirements.txt`; their pretrained OCR assets may initialize on first use.
- Club scans can fail with glare, motion blur, an obstructed sole/face, a tiny marking, a reflective finish, or a frame that is too wide.
- Driver, Wood, and Hybrid do not run the Iron/Wedge OCR stage.

## Movement analysis and KPIs

### Local pose pipeline

`ModelManager` validates that OpenCV can open the uploaded video, records basic video metadata, extracts fallback frames, and attempts local pose inference.

When Ultralytics and a compatible YOLO pose model are available, the pose estimator maps COCO keypoints into named landmarks, including head, shoulders, elbows, wrists, hips, knees, and ankles. The model loader checks a configured path, `models/pretrained/yolov8n-pose.pt`, and `yolov8n-pose.pt` in the working directory.

The pipeline then:

- Writes pose landmark rows to `outputs/experiments/pose_landmarks.csv` when landmarks are found.
- Calculates player-facing and supporting movement measurements.
- Builds raw and smoothed pose-overlay videos.
- Validates generated overlays and preserves tracking-quality metadata.
- Stores warnings and fallback status if usable pose evidence or an overlay is unavailable.

### KPI reference

| KPI | Meaning | Source |
| --- | --- | --- |
| Overall Swing Score | A local or Gemini-supplied 0-100 summary when enough evidence is available. | Coaching engine and optional Gemini structured result |
| Kinematic Sequence | The estimated order and spacing of hip, torso, and arm movement peaks. | Pose-frame movement peaks |
| X-Factor Separation | A 2D proxy for the shoulder/hip rotational separation during the swing. | Pose-frame shoulder and hip axes |
| Spine Angle Maintenance | How consistently the estimated spine angle is maintained. | Pose-frame spine-angle variation |
| Lateral Weight Shift | A 2D estimate of lateral hip movement relative to stance width. | Pose-frame hip centers and stance width |

The UI classifies available KPI scores as Strong, On track, Developing, Needs work, or More data. The PDF uses a closely aligned status scale:

| Score | PDF status |
| --- | --- |
| 80-100 | Strong |
| 65-79 | On Track |
| 50-64 | Developing |
| Below 50 | Needs Focus |

These are video-derived practice signals. They are not laboratory-grade biomechanics, ball-flight measurements, club speed, launch, contact, or injury diagnostics.

### Other movement measurements

Depending on available landmarks, SwingSight can calculate or retain:

- Spine angle and spine-angle variation.
- Shoulder rotation proxy / estimated shoulder turn.
- Hip rotation proxy.
- Head, hip, knee, and foot movement.
- Hand-path distance.
- Knee flex.
- Lateral-shift ratio.
- Balance, tempo, and posture-change proxies.

Many of these are 2D video estimates. A value can be absent when landmarks are unavailable, and a zero can be ambiguous. The professional PDF filters questionable data; the Advanced details panel retains the raw local result for debugging.

## Optional Gemini coaching

Gemini is optional. Without a key, SwingSight still runs local analysis and baseline coaching.

When configured, `GeminiCoachingService` sends a structured JSON evidence payload containing selected local metrics, tracking-quality summary, club label, and which fields were unavailable. It intentionally excludes:

- Source video and uploaded media.
- Raw frames and pose-landmark files.
- Local file paths.
- Debug artifacts and overlay files.

Gemini is asked to return validated JSON with a summary, an eligible 0-100 score or `null`, score rationale, next focus, up to three strengths, up to three improvements, tips, drills, and data gaps. The service rejects invalid responses and falls back to local coaching when the key is absent, the request fails, or the response does not match the schema.

### Configure Gemini

Create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_key_here
```

The loader accepts UTF-8 and UTF-8-with-BOM `.env` files. It reads the configured environment variable first, then `GEMINI_API_KEY` and `GOOGLE_API_KEY` as fallbacks.

The example configuration defaults to:

```yaml
gemini:
  enabled: true
  api_key_env: GEMINI_API_KEY
  model: gemini-3-flash-preview
  timeout_seconds: 35
  max_output_tokens: 4096
```

Never commit a `.env` file or a real API key. Rotate a key immediately if it is exposed.

## Install and run

### Requirements

- A supported Python 3 interpreter.
- A modern browser with camera permission for live capture.
- Internet access only for first-time dependency installation, optional Gemini coaching, and any first-use OCR/model downloads required by installed runtimes.
- Enough local disk space for Python packages, model weights, temporary videos, generated overlays, JSON results, and PDF reports.

Core runtime dependencies are declared in `requirements.txt`:

- Flask and Python dotenv handling.
- OpenCV, MediaPipe, Ultralytics, PyTorch, and Torchvision.
- NumPy, pandas, scikit-learn, Pillow, and PyYAML.
- RapidOCR and ONNX Runtime for exact club markings.
- ReportLab for PDF reports.
- Requests for optional Gemini calls.
- Pytest for automated checks.

### Windows launcher

1. Clone or download this repository.
2. Double-click **Launch SwingSight.bat**.
3. The launcher finds Python 3, creates `.venv` if needed, upgrades `pip`, installs `requirements.txt`, enables browser opening, and starts `src/run.py`.
4. Open `http://127.0.0.1:8000` if the browser does not open automatically.

The Windows launcher checks and installs requirements on every start, so requirements changes are applied automatically.

### macOS launcher

1. Clone or download this repository.
2. Double-click **SwingSight.app**.
3. The app opens Terminal and runs `scripts/start-macos.sh`.
4. The script creates `.venv` and installs dependencies on the first run, then starts the local server.
5. Open `http://127.0.0.1:8000` in a browser if it does not open automatically.

macOS may ask for permission to open the application or Terminal. The launcher requires `python3` to be available on `PATH`.

### Terminal launch: Windows, macOS, or Linux

```bash
# From the repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/run.py
```

The server defaults to `127.0.0.1:8000`. Stop it with `Ctrl+C`.

### Runtime environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWINGSIGHT_HOST` | `127.0.0.1` | Flask bind host. Keep the loopback default for local use. |
| `SWINGSIGHT_PORT` | `8000` | Flask port. |
| `SWINGSIGHT_DEBUG` | `true` | Enables Flask debug behavior unless set to `0`, `false`, or `no`. |
| `SWINGSIGHT_OPEN_BROWSER` | unset | Opens a browser tab when set to `1`, `true`, `yes`, or `on`. The Windows launcher sets it. |
| `GEMINI_API_KEY` | unset | Optional Gemini credential. |
| `GOOGLE_API_KEY` | unset | Optional fallback Gemini credential. |

Example:

```powershell
$env:SWINGSIGHT_PORT = "8001"
$env:SWINGSIGHT_DEBUG = "false"
python src/run.py
```

## Configuration

`swingsight.config.load_config()` uses this order:

1. Built-in defaults in `src/swingsight/config.py`.
2. `config.yaml` in the repository root, when present.
3. Otherwise `config.example.yaml`.

The selected YAML file is deep-merged onto the built-in defaults, so a partial `config.yaml` only needs to contain values you want to change.

### Important configuration groups

| Group | What it controls |
| --- | --- |
| `paths` | Local `data`, `models`, `uploads`, `outputs`, and `reports` directories. |
| `pose_estimation` | Pose confidence and optional model path. |
| `club_localization` | Hand-based ROI crop, MediaPipe hand model path, confidence, crop scale, and merge behavior. |
| `club_recognition` | Five-way checkpoint, confidence thresholds, and legacy checkpoint paths. |
| `club_recognition.marking_ocr` | OCR backend, confidence threshold, image sizing, contrast, sharpening, and sideways fallback passes. |
| `metrics` | Metric smoothing and normalization settings. |
| `gemini` | Gemini enablement, API-key environment variable, model, timeout, and output budget. |
| `preprocessing` | Adaptive-contrast behavior for frames and club images. |
| `feedback` | Per-club baseline feedback thresholds. |
| `app` | Local host/port, key-frame stride, and maximum-video target. |

Copy the example before changing it:

```bash
cp config.example.yaml config.yaml
```

On Windows PowerShell:

```powershell
Copy-Item config.example.yaml config.yaml
```

### Example customizations

```yaml
club_recognition:
  five_way_cnn_min_confidence: 0.70
  marking_ocr:
    min_confidence: 0.75

gemini:
  enabled: false

app:
  local_web_port: 8001
```

`src/run.py` currently reads the Flask host and port from the `SWINGSIGHT_HOST` and `SWINGSIGHT_PORT` environment variables, so set those variables when changing the actual development-server bind address.

## Local API

The browser uses the following local endpoints. They are not authenticated public APIs and should remain behind the local loopback server unless you add appropriate production security.

| Method | Endpoint | Input | Result |
| --- | --- | --- | --- |
| `GET` | `/` | None | Landing and capture page. |
| `GET` | `/analysis/<analysis_id>` | Path ID | Completed-review page shell. |
| `POST` | `/api/upload-video` | Multipart field `video` | Saves a video and returns `upload_id`, local file name, and preview URL. |
| `POST` | `/api/upload-club-image` | Multipart field `image` | Saves a club image and returns its upload metadata. |
| `POST` | `/api/record-swing` | Multipart field `video` | Saves a browser-recorded clip and returns upload metadata. |
| `POST` | `/api/club-detect` | Multipart field `frame` | Runs club recognition on a still frame. |
| `POST` | `/api/body-check` | Multipart field `frame` | Checks whether key body landmarks are visible. |
| `POST` | `/api/analyze` | JSON described below | Analyzes an uploaded video, optionally with a separate club image. |
| `POST` | `/api/analyze-swing` | JSON described below | Analyzes a recorded swing using the recorded club category. |
| `GET` | `/api/results/<analysis_id>` | Path ID | Returns the full persisted result. |
| `GET` | `/api/artifacts/<analysis_id>` | Path ID | Returns overlay files and advanced metrics. |
| `GET` | `/api/analysis-assets/<analysis_id>` | Path ID | Returns overlay files, advanced metrics, and summary. |
| `POST` | `/api/reports/<analysis_id>` | None | Generates the PDF and returns its local download URL. |
| `GET` | `/uploads/<filename>` | Path | Serves a saved upload. |
| `GET` | `/outputs/<filename>` | Path | Serves a generated overlay/output. |
| `GET` | `/reports/<filename>` | Path | Downloads a PDF report only. Non-PDF report requests return 404. |

### Analysis request bodies

```json
{
  "video_upload_id": "returned-by-upload-video",
  "club_category": "8 Iron",
  "club_image_upload_id": "optional-upload-club-image-id"
}
```

`/api/analyze-swing` accepts `video_upload_id` and `club_category`; it does not require `club_image_upload_id` because the live flow performs its club scan before recording.

Successful analysis results include fields such as:

```json
{
  "status": "success",
  "analysis_id": "...",
  "club": "8 Iron",
  "swing_score": 78,
  "score_label": "Improving",
  "next_focus": "Keep a smooth tempo through impact.",
  "strengths": [],
  "improvements": [],
  "advanced_metrics": {},
  "tracking": {},
  "visualization": {},
  "overlay_files": [],
  "warnings": []
}
```

The exact fields are intentionally richer than the player-facing UI. `advanced_metrics`, `tracking`, `model_outputs`, `debug`, and related fields are useful for development; do not treat them as stable third-party API contracts without versioning them first.

## Storage, privacy, and security

### Local files

| Location | Contents |
| --- | --- |
| `uploads/` | Browser uploads, recorded clips, club images, and captured frames. |
| `outputs/` | Persisted `analysis_<id>.json` results, overlays, experiments, pose landmarks, fallback frames, and diagnostics. |
| `reports/` | Generated PDF coaching reports. |
| `data/` | Raw and processed local datasets used by training/experiments. |
| `models/` | Local trained or reference model checkpoints. |

These directories can contain personal videos and derived movement data. Keep them out of public repositories and back them up or delete them according to your own privacy policy. The repository's `.gitignore` should continue to exclude local media, virtual environments, credentials, and large experimental data.

### Gemini boundary

When Gemini is enabled, SwingSight transmits structured local evidence to the Gemini API. It does **not** intentionally transmit the uploaded swing video, raw frames, local file paths, or debug artifacts. Review the running configuration and your network environment before using Gemini with sensitive data.

### Local server warning

The bundled Flask server is for local development. It does not provide production authentication, rate limiting, multi-user isolation, encrypted remote transport, or hardened file access. Do not expose it directly to the internet.

## Models, data, and training

### Bundled and expected runtime assets

| Asset | Purpose |
| --- | --- |
| `models/trained/club_type_5way.pt` | Preferred ResNet-50 five-way club classifier. |
| `models/trained/club_type_5way_cnn.pt` | Compact five-way CNN reference/fallback checkpoint when available. |
| `yolov8n-pose.pt` | Local YOLO pose candidate used by the pose runtime. |
| `hand_landmarker.task` | MediaPipe hand-landmark asset used for optional club-centered cropping. |
| RapidOCR PP-OCR models | Pretrained OCR assets used for exact Iron/Wedge readings. |

The five-way loader validates that a checkpoint is self-describing: it expects the SwingSight checkpoint format, task name, class names, input size, normalization values, and model weights. It supports the current ResNet-50 format and compatible earlier MobileNetV3-Small checkpoints.

Legacy broad-category, Iron-number, and wood-type checkpoints remain supported for older installations, but the preferred live workflow is the five-way checkpoint plus RapidOCR for Iron/Wedge markings.

### Training data layout

```text
data/club_training/
  images/                         # original club images
  annotations/
    club_manifest.csv
  derived/                        # notebook-generated material
```

The historical manifest format is:

```text
image_path,split,five_way_label,marking_label,mark_x,mark_y,mark_w,mark_h
```

- `image_path` is relative to `data/club_training/images`.
- `split` is normally `train` or `val`.
- `five_way_label` is `driver`, `wood`, `hybrid`, `iron`, or `wedge`.
- The `marking_*` columns are retained for historical experiments; production exact-marking recognition uses pretrained OCR instead of a custom marking model.

Keep images from one source capture in a single split. Similar augmented images across train and validation sets can leak near-duplicates and make validation accuracy misleading.

### Notebooks and scripts

| Path | Purpose |
| --- | --- |
| `notebooks/01_golfdb_video_model_training.ipynb` | Golf-swing video research and training exploration. |
| `notebooks/02_authorized_driver_dataset_builder.ipynb` | Builds authorized driver-head datasets from approved direct-URL manifests; it does not scrape the USGA database. |
| `notebooks/03_train_five_way_club_cnn.ipynb` | Trains the ResNet-50 five-way club checkpoint. |
| `scripts/train_club_cnn.py` | Script support for club-CNN training. |

The five-way ResNet-50 training workflow is intended for a CUDA-capable GPU. Inference can run locally on a compatible CPU or GPU runtime, subject to installed packages and model availability.

### Reference images

`assets/club-reference-images/` contains openly licensed real golf-club photographs organized by recognition taxonomy. They are reference material, not a production training dataset. See that directory's README for source and license details.

## Project structure

```text
SwingSight-AI/
├── Launch SwingSight.bat          # Windows entry point
├── SwingSight.app/                # macOS Terminal launcher
├── config.example.yaml            # configuration reference
├── requirements.txt               # runtime and test dependencies
├── hand_landmarker.task           # hand-crop model asset
├── yolov8n-pose.pt                # local pose model candidate
├── src/
│   ├── run.py                     # local Flask entry point
│   ├── backend/services/
│   │   ├── model_manager.py       # video validation, pose/club pipeline, overlays
│   │   ├── coaching_engine.py     # local coach-style feedback
│   │   └── overlay_generator.py   # raw/smoothed overlay output
│   ├── swingsight/
│   │   ├── metrics.py             # 2D movement and KPI calculations
│   │   ├── pose_estimation.py     # YOLO keypoint mapping
│   │   ├── club_recognition.py    # five-way and exact club workflow
│   │   ├── club_marking_ocr.py    # RapidOCR integration
│   │   ├── club_localization.py   # hand-centered ROI crop
│   │   └── config.py              # defaults, YAML, and .env loading
│   └── webapp/
│       ├── routes/dashboard.py    # browser and JSON routes
│       ├── services/              # orchestration, Gemini, PDF report
│       ├── templates/             # landing and review pages
│       └── static/                # CSS and browser JavaScript
├── models/                        # checkpoints and model documentation
├── data/                          # local data and training inputs
├── uploads/                       # runtime media uploads
├── outputs/                       # results, overlays, experiments, diagnostics
├── reports/                       # generated PDF reports
├── notebooks/                     # research and training notebooks
├── scripts/                       # launch/training support
└── tests/                         # automated unit and smoke checks
```

## Development and testing

### Run tests

```bash
PYTHONPATH=src pytest -q
```

Useful focused commands:

```bash
# PDF generation and download behavior
PYTHONPATH=src pytest -q tests/test_reports.py

# Club recognition and OCR normalization
PYTHONPATH=src pytest -q tests/test_club_recognition.py tests/test_club_marking_ocr.py

# Basic movement-metric smoke test
PYTHONPATH=src pytest -q tests/test_smoke.py
```

The tests cover core metric keys, club classifier/OCR branching and normalization, model-manager club replacement behavior, Gemini schema handling, and PDF generation/download/formatting helpers.

If an active project-root `.env` contains a real Gemini key, isolate or temporarily move it before running offline Gemini tests. The Gemini service deliberately reloads the project `.env`, so an active key can cause a test intended to verify the no-key path to attempt a network request.

### Working on the frontend

- Landing-page behavior: `src/webapp/templates/dashboard.html`, `src/webapp/static/js/dashboard.js`, and `src/webapp/static/css/dashboard.css`.
- Review-page behavior: `src/webapp/templates/analysis.html` and `src/webapp/static/js/analysis.js`.
- Templates use cache-busting version parameters on CSS/JS links. Bump the relevant parameter after an asset change when a browser keeps an older file.
- Keyboard focus styling, reduced-motion handling, responsive stacking, and local-only capture are part of the current UI implementation.

### Working on the report

`src/webapp/services/report_service.py` owns the PDF output. Its helpers cover safe numeric conversion, validation, score/degree/pixel/ratio formatting, metric labels, text sanitization and wrapping, short report IDs, and club-aware filenames. Keep the report data-driven; do not expose raw analysis JSON, full UUIDs, non-finite values, or unbounded text in player-facing sections.

## Troubleshooting

### Python or dependencies are missing

Install a current Python 3 release, then rerun the launcher or recreate `.venv` and install `requirements.txt`. On Windows, the launcher searches `py -3` first and then `python`.

### The browser does not open

Visit `http://127.0.0.1:8000` manually. Set `SWINGSIGHT_OPEN_BROWSER=true` when launching from a terminal if you want `src/run.py` to open a tab.

### The dashboard shows old CSS or JavaScript

Hard refresh the browser:

- Windows: `Ctrl+F5` or `Ctrl+Shift+R`
- macOS: `Cmd+Shift+R`

Then restart SwingSight. Check the cache-busting asset version in the relevant template if the issue persists.

### The camera cannot open

Use HTTPS or `localhost`/`127.0.0.1`, grant camera permission to the browser, close other apps using the camera, and try the upload path if live capture remains unavailable.

### Club recognition is unavailable or uncertain

Confirm that `models/trained/club_type_5way.pt` exists and that PyTorch/Torchvision can load it. Show the club face or sole close to the camera, reduce glare, hold it steady, and avoid cropping the club head out of frame. Iron/Wedge OCR needs a readable marking; it is normal for the result to remain the broad club family when a number, letter, or loft cannot be confirmed.

### Pose tracking or overlay generation fails

Use a short, readable video with the full body in frame. Confirm that OpenCV can open the format and that the YOLO pose model and Ultralytics runtime are available. Check the Advanced details section and `outputs/` for warnings, tracking metadata, and generated assets.

### Windows OpenH264 or VideoWriter errors

Messages such as the following usually mean OpenCV/FFmpeg could not initialize an H.264 encoder:

```text
Failed to load OpenH264 library
VIDEOIO/FFMPEG: Failed to initialize VideoWriter
```

The analysis may still finish, but an overlay can be absent or incomplete. Verify browser playback, use a compatible OpenCV/FFmpeg build or trusted local encoder, and ensure any native library matches the installed Python/OpenCV architecture. Do not download arbitrary DLLs from untrusted sources.

### Gemini is not configured

Add `GEMINI_API_KEY` to the project-root `.env`, confirm the filename is exactly `.env` rather than `.env.txt`, restart the app, and do not put spaces around the equals sign. The app remains usable without Gemini.

### The PDF report is missing

Open the completed review and use **Download PDF Report**. Check that `reports/` is writable and inspect the local app logs for ReportLab errors. Only `.pdf` files are served from the reports route.

## License and contributions

SwingSight AI is an evolving project. Add an explicit license, contribution policy, data-governance policy, and production security design before redistributing, deploying publicly, or accepting external contributions.
