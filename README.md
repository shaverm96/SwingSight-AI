# SwingSight AI

A local golf-swing coach that analyzes uploaded or recorded video, creates a motion overlay, identifies the club when available, and provides practice feedback.

## Launch SwingSight

| macOS | Windows |
| --- | --- |
| Double-click **SwingSight.app** | Double-click **Launch SwingSight.bat** |

Both launchers open Terminal or Command Prompt, create a local Python environment on first use, install the app, and start the local dashboard at <http://127.0.0.1:8000>.

For detailed Gemini feedback, add `GEMINI_API_KEY=...` to a local `.env` file. The key is never committed.

## Project map

- `SwingSight.app` / `Launch SwingSight.bat` — the only supported double-click launchers.
- `src/` — application code: web UI, analysis, vision, scoring, and reports.
- `data/` — local sample and working video data.
- `models/` — local model checkpoints; keep trained files in `models/trained/`.
- `outputs/`, `reports/`, `uploads/` — files created while the app runs.
- `tests/` — automated checks.
- `notebooks/`, `assets/`, and `scripts/` — optional model-training, research, and launcher support files.

## What the app does

1. Upload or record a swing.
2. Track body motion and build an overlay.
3. Use the selected club, or detect it during guided capture.
4. Combine local vision measurements with Gemini coaching when configured.
5. Show a score, priorities, practice cues, and downloadable reports.

## Windows notes

The Windows launcher first uses `.venv\Scripts\python.exe`. If it does not exist, it tries the Windows Python launcher (`py -3`) and then `python` to create it. If neither is installed, it leaves a clear message with the Python download link.

The app is designed to run on Windows 10/11 and macOS. A compatible GPU build of PyTorch is optional; the default install boots with the portable CPU build.

## Development

Run the focused test suite from the project folder:

```bash
python -m pytest -q
```

The research notebooks and training script are intentionally separate from the app runtime. They are not needed to launch or use SwingSight.
