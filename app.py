from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from webapp import create_app


app = create_app()


if __name__ == "__main__":
    # Local-only development server. Keep this app off public networks.
    app.run(host="127.0.0.1", port=8000, debug=True)
