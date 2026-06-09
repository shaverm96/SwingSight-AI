# SwingSight AI Platform Compatibility Checklist

## Supported Platforms

- macOS on Apple Silicon
- macOS on Intel
- Windows 10
- Windows 11

## Startup

- [ ] Clone the repository
- [ ] Create a Python virtual environment
- [ ] Install dependencies with `pip install -r requirements.txt`
- [ ] Start the app with `start.sh`, `start.bat`, or `python run.py`
- [ ] Confirm the app opens on `http://localhost:8000`

## Camera

- [ ] Browser camera permission prompt appears
- [ ] Webcam discovery works without hardcoded camera indices
- [ ] Default camera fallback works if a selected camera is unavailable
- [ ] Mac webcam capture works
- [ ] Windows webcam capture works
- [ ] External USB camera capture works

## Analysis

- [ ] Video upload works
- [ ] YOLOv8 pose detection works on CPU
- [ ] YOLOv8 pose detection uses CUDA on Windows when available
- [ ] YOLOv8 pose detection uses MPS on Apple Silicon when available
- [ ] Overlay generation completes successfully
- [ ] Swing analysis completes successfully

## Exports

- [ ] PDF export works
- [ ] Word export works
- [ ] Body landmark coordinate CSV is created
- [ ] Wide coordinate CSV is created
- [ ] Output files are saved under relative project paths

## Browser Validation

- [ ] Chrome works on macOS
- [ ] Chrome works on Windows
- [ ] Edge works on Windows
- [ ] Safari works on macOS
- [ ] Firefox works on both platforms
