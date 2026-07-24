# SwingSight AI

SwingSight AI is a local-first golf swing-analysis application. It turns a short uploaded or recorded swing video into a motion review, five key performance indicators (KPIs), an overall Swing Score, and practical coach-style feedback.

The application is built for real practice sessions: run it on your computer, upload a clear swing video, and review the result in your browser. Gemini coaching is optional; local video analysis and local coaching continue to work without an API key.

> **Project status:** Active development. SwingSight supports practice and video review; it is not a substitute for an in-person PGA professional, medical, or injury assessment.

## Highlights

- Local web dashboard for swing upload and camera capture
- Pose-based movement analysis with a rendered swing overlay
- Five golfer-friendly KPI rankings: Overall Swing Score, Kinematic Sequence, X-Factor Separation, Spine Angle Maintenance, and Lateral Weight Shift
- Coach-style feedback with one priority and three detailed observations
- Optional Gemini-enhanced narrative coaching based on local CV measurements
- Club-recognition support for Driver, Wood, Hybrid, Iron, and Wedge
- Optional exact iron/wedge marking recognition after the club head is localized
- Windows launcher that creates a virtual environment, installs requirements, and opens the application in the default browser
- Training notebooks that use one shared club-image dataset for club-head detection and exact club-marking recognition

## Quick start

### Windows

1. Download or clone this repository.
2. Double-click **Launch SwingSight.bat**.
3. On first launch, SwingSight locates Python 3, creates a .venv environment, installs the packages in requirements.txt, and starts the dashboard.
4. It opens the dashboard in your default browser. If it does not, visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

The launcher checks packages on every start, so changes to requirements.txt are automatically applied.

If it cannot find Python, install a current Python 3 release from [python.org](https://www.python.org/downloads/), then run the launcher again.

### macOS

1. Download or clone this repository.
2. Double-click **SwingSight.app**.
3. Allow the application if macOS requests permission, then open [http://127.0.0.1:8000](http://127.0.0.1:8000) if needed.

### Terminal launch

Use a terminal when developing or troubleshooting.

~~~bash
# From the repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/run.py
~~~

The site uses http://127.0.0.1:8000 by default. Stop it with Ctrl+C.

## Configure optional Gemini coaching

SwingSight uses local computer-vision measurements by default. Configure Gemini only when you want expanded narrative coaching.

Create a file named .env in the project root, beside README.md:

~~~dotenv
GEMINI_API_KEY=your_key_here
~~~

Restart SwingSight after saving the file. The loader accepts UTF-8 and UTF-8-with-BOM .env files, which helps avoid a common Windows Notepad encoding issue.

To confirm the current environment can see the key:

~~~powershell
.venv\Scripts\python.exe -c "from pathlib import Path; import os, sys; sys.path.insert(0, 'src'); from swingsight.config import load_dotenv; load_dotenv(Path('.env')); print('Gemini key detected' if os.getenv('GEMINI_API_KEY') else 'Gemini key not detected')"
~~~

If it is not detected, verify that:

- The filename is exactly .env, not .env.txt.
- The file is in the repository root.
- The value is a single line in the form GEMINI_API_KEY=value, without spaces around the equals sign.
- You restarted SwingSight after editing the file.

Never commit a .env file or share an API key. Rotate a key immediately if it is exposed.

## Using SwingSight

1. Start the app and choose **Record new swing** or upload a video.
2. Record from a clear, consistent side angle with your full body, hands, club, and feet in frame.
3. Keep the camera still and avoid zooming.
4. Submit the swing and wait for processing.
5. Review the original video, motion overlay, Swing Score, KPI cards, Coach's Priority, and practice cues.

### Recording recommendations

For the most useful review:

- Use one swing per clip.
- Keep the full swing in frame from setup through follow-through.
- Use good lighting and a contrasting background where possible.
- Keep the camera stable and roughly hip-high.
- Use a comparable camera angle for each progress check.
- Center the club head in the frame if club recognition matters.

## KPI reference

| KPI | What it measures |
| --- | --- |
| **Overall Swing Score** | A 0–100 summary of the available movement, posture, timing, and balance signals. It is most useful for comparing similar recordings over time. |
| **Kinematic Sequence** | How efficiently the pelvis, torso, arms, and club appear to accelerate and decelerate in order. |
| **X-Factor Separation** | The rotational difference between the hips and shoulders near the top of the backswing—the coil that can contribute to speed. |
| **Spine Angle Maintenance** | How consistently posture and spinal tilt are held from address through impact. |
| **Lateral Weight Shift** | How body position and pressure move toward the lead side while avoiding excessive sway or thrust. |

Status labels are color coded in the dashboard:

- **On Track / Strong** — green
- **Developing** — amber
- **Needs Work / Needs Practice** — red
- **More Data** — gray

These are directional coaching signals, not laboratory-grade biomechanics measurements.

## Club recognition and models

SwingSight uses a staged recognition workflow:

~~~text
Club-head detection
       ↓
Five-way classification: Driver | Wood | Hybrid | Iron | Wedge
       ↓
Optional exact marking: 1–9, P/A/G/S/L, or a wedge loft
~~~

The app can complete the swing review even when club recognition is uncertain. It reports the missing capability rather than falsely confirming a club.

Expected optional local model files:

~~~text
models/
  club_detector.pt                  # club-head detector
  trained/
    club_type_5way.pt               # driver / wood / hybrid / iron / wedge
    club_marking_cnn.pt             # exact iron/wedge marking
~~~

Messages such as **club-head detector unavailable** or **club_marking CNN checkpoint was not found** mean the related optional model has not been added yet. The rest of SwingSight can still analyze the swing, but club identification will be less precise.

## Training club models

The training workflow deliberately keeps one master image library instead of duplicating training images per model.

~~~text
data/club_training/
  images/
    ... original club images ...
  annotations/
    club_manifest.csv
  derived/                          # generated by notebooks
~~~

The manifest uses one row per image:

~~~text
image_path,split,five_way_label,head_x,head_y,head_w,head_h,marking_label,mark_x,mark_y,mark_w,mark_h
~~~

- **image_path:** path relative to data/club_training/images
- **split:** train or val
- **five_way_label:** driver, wood, hybrid, iron, or wedge
- **head_***: normalized club-head bounding box for the detector
- **marking_***: normalized bounding box around the readable number, letter, or loft
- **marking_label:** one of 1–9, p/a/g/s/l, or loft labels 50/52/54/56/58/60

Run the notebooks in order:

1. **notebooks/02_train_club_head_detector.ipynb** creates models/club_detector.pt.
2. **notebooks/03_train_club_marking_cnn.ipynb** crops annotated markings from the shared master images and creates models/trained/club_marking_cnn.pt.

Keep images from a single source capture in one split only. Otherwise, nearly identical images can appear in both training and validation, which gives misleadingly strong results.

## Configuration

Use config.example.yaml as the reference for model paths, thresholds, processing limits, and Gemini settings.

| Setting | Default |
| --- | --- |
| Local host | 127.0.0.1 |
| Local port | 8000 |
| Maximum video duration | 12 seconds |
| Pose backend | MediaPipe |
| Gemini environment variable | GEMINI_API_KEY |
| Club detector path | models/club_detector.pt |

For example, run a development server on a different port:

~~~powershell
$env:SWINGSIGHT_PORT = "8001"
$env:SWINGSIGHT_DEBUG = "false"
python src/run.py
~~~

## Project structure

~~~text
SwingSight-AI/
├── Launch SwingSight.bat          # Windows launcher
├── SwingSight.app/                # macOS launcher
├── config.example.yaml            # configuration reference
├── requirements.txt               # runtime and test dependencies
├── src/
│   ├── run.py                     # Flask entry point
│   ├── backend/                   # analysis and coaching services
│   ├── swingsight/                # configuration and vision utilities
│   └── webapp/                    # dashboard routes, templates, and assets
├── notebooks/                     # model-training notebooks
├── scripts/                       # launcher and training support
├── models/                        # reference and locally trained checkpoints
├── data/                          # local working and training data
├── uploads/                       # videos created at runtime
├── outputs/                       # overlays and analysis outputs
├── reports/                       # generated reports
└── tests/                         # automated checks
~~~

Videos, generated outputs, model checkpoints, virtual environments, and .env files should stay local unless you explicitly intend to version them.

## Development and testing

Install project dependencies, then run the focused test suite from the repository root:

~~~bash
python -m pytest -q
~~~

Useful commands:

~~~bash
# Run the local app
python src/run.py

# Run a specific test module
python -m pytest tests/<test_file>.py -q

# Confirm the active interpreter
python --version
~~~

The bundled Flask server is for local development. Do not expose it directly to the internet as a production deployment.

## Troubleshooting

### Python is not found

Install Python 3 from [python.org](https://www.python.org/downloads/), reopen Command Prompt or PowerShell, and launch SwingSight again.

### Browser does not open automatically

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) manually. If you set SWINGSIGHT_PORT, use that port instead.

### Gemini key is not configured

Follow the **Configure optional Gemini coaching** section. The key must be in the project-root .env file, and SwingSight must be restarted after every change.

### A club model is missing

This limits club-recognition precision but does not block regular video analysis. Train the models using the notebooks above or place validated model files in the configured locations.

### OpenH264 or VideoWriter errors on Windows

Errors like:

~~~text
Failed to load OpenH264 library
VIDEOIO/FFMPEG: Failed to initialize VideoWriter
~~~

mean that OpenCV/FFmpeg could not initialize an H.264 encoder. The analysis may still finish, but an overlay video can be absent or incomplete.

First verify the overlay output and browser playback. If overlay videos consistently fail, use a compatible OpenCV/FFmpeg build or a trusted local H.264 encoder. Any DLL must match the installed Python and OpenCV architecture, normally 64-bit on modern Windows. Do not download arbitrary DLLs from untrusted sources.

### The dashboard still shows old styling after an update

Hard refresh the browser:

- **Windows:** Ctrl+F5 or Ctrl+Shift+R
- **macOS:** Cmd+Shift+R

Then restart SwingSight.

## Privacy and security

- SwingSight is intended to run locally on your computer.
- Uploaded videos, generated overlays, reports, and model artifacts are local project data.
- With Gemini enabled, the default configuration is designed to send structured local coaching measurements—not the source video, raw frames, file paths, or debug artifacts.
- Keep .env files, API keys, and personal swing videos out of public repositories.

## License and contributions

SwingSight AI is an evolving project. Add an explicit license and contribution policy before redistributing, deploying, or accepting external contributions.
