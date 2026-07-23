from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(ROOT)

from swingsight.config import load_dotenv

load_dotenv(ROOT / ".env")

from webapp import create_app


def _is_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _open_browser(url: str, debug: bool) -> None:
    """Open once from the process that serves the Flask app."""
    if not _is_enabled(os.environ.get("SWINGSIGHT_OPEN_BROWSER")):
        return
    if debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    timer = threading.Timer(1.0, lambda: webbrowser.open_new_tab(url))
    timer.daemon = True
    timer.start()


def main() -> None:
    app = create_app()
    host = os.environ.get("SWINGSIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("SWINGSIGHT_PORT", "8000"))
    debug = os.environ.get("SWINGSIGHT_DEBUG", "true").lower() not in {"0", "false", "no"}
    _open_browser(f"http://{host}:{port}", debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
